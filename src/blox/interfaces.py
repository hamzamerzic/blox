"""Core interfaces and abstractions for the blox library.

This module defines the fundamental building blocks of the library, including:
- Graph: Structural definition of the model hierarchy.
- Params: Immutable, functional state container.
- Module: Base class for neural network layers.
- SequenceBase/RecurrenceBase: Interfaces for sequence processing.

It is designed to be strictly typed and utilizes chex for runtime shape checking.
"""

from __future__ import annotations
import inspect
from typing import Any, Callable, Generic, TypeVar, cast

import chex
import jax
import jax.numpy as jnp

# ==============================================================================
# Type Definitions
# ==============================================================================

Shape = tuple[int, ...]
Initializer = jax.nn.initializers.Initializer
Path = tuple[str, ...]

InputsT = TypeVar('InputsT', bound=chex.ArrayTree)
StateT = TypeVar('StateT', bound=chex.ArrayTree)
OutputsT = TypeVar('OutputsT', bound=chex.ArrayTree)
ResetT = TypeVar('ResetT', bound=chex.ArrayTree)


# ==============================================================================
# Graph & Param
# ==============================================================================


class Graph:
  """The structural graph of a model.

  A Graph represents the hierarchical structure of your model. Each node in the
  graph corresponds to a module (layer), and edges represent the parent-child
  relationships between them. When you create a child node with `graph.child()`,
  you're extending this structure.

  The graph serves two purposes:
  1. It defines how your model is organized - which modules contain which.
  2. It provides unique namespaces for parameters. Each node's path (e.g.,
     ('net', 'encoder', 'dense')) becomes the prefix for that module's params.

  Dependency injection creates additional relationships in the graph. When a
  module is created externally and passed into another, it retains its original
  position in the graph (as a sibling rather than a child), enabling flexible
  parameter sharing patterns.

  The graph does not store parameters - that's the job of the Params container.
  Graph defines structure; Params holds state.
  """

  def __init__(self, name: str) -> None:
    """Initializes a graph node.

    Args:
      name: The name of this node. Must not be empty.

    Raises:
      ValueError: If the name is empty.
    """
    if not name:
      raise ValueError('Graph node must have a name.')
    self.name = name
    self.path: Path = (name,)
    self._children: dict[str, Graph] = {}
    # Metadata storage for visualization or auxiliary info.
    self.metadata: dict[str, Any] = {}
    # Track if this node is the root of the hierarchy.
    self._is_root = True

  def child(self, name: str) -> Graph:
    """Creates or retrieves a child node in the graph hierarchy.

    Args:
      name: The name of the child node.

    Returns:
      A new Graph instance representing the child.

    Raises:
      ValueError: If a child with the same name already exists.
    """
    if name in self._children:
      raise ValueError(
        f"Graph node '{self.path}' already has a child named '{name}'."
      )
    child_node = Graph(name)
    child_node._set_parent(self)
    self._children[name] = child_node
    return child_node

  def __truediv__(self, name: str) -> Graph:
    """Syntactic sugar for creating children using the '/' operator.

    This is semantically equivalent to calling `self.child(name)`.

    Example:
        # These are identical:
        sub = graph / 'layer1'
        sub = graph.child('layer1')

    Args:
      name: The name of the child node.

    Returns:
      A new Graph instance representing the child scope.
    """
    return self.child(name)

  def _set_parent(self, parent: Graph) -> None:
    self.path = parent.path + (self.name,)
    # This node is now part of a hierarchy, so it is no longer a root.
    self._is_root = False

  def __repr__(self) -> str:
    n_children = len(self._children)
    path_str = '/'.join(self.path)
    if n_children == 0:
      return f"Graph('{path_str}')"
    return f"Graph('{path_str}', children={n_children})"


class Param:
  """A wrapper around a parameter value that holds metadata.

  Attributes:
    value: The actual JAX array or PyTree stored.
    trainable: Boolean flag indicating if gradients should be computed.
    metadata: Dictionary for arbitrary tags. Common keys include:
      - 'sharding': tuple of axis names (e.g., (None, 'model')) for partitioning
      - 'tag': string identifier (e.g., 'rng', 'optimizer_state')
  """

  def __init__(
    self,
    value: Any,
    trainable: bool = True,
    metadata: dict[str, Any] | None = None,
  ) -> None:
    self.value = value
    self.trainable = trainable
    self.metadata = metadata or {}

  @property
  def sharding(self) -> tuple[str | None, ...]:
    """Returns the sharding spec from metadata, if present."""
    return self.metadata.get('sharding', ())

  def replace(self, **updates: Any) -> Param:
    """Creates a new Param with updated fields.

    Args:
      **updates: Keyword arguments matching the attribute names to update.

    Returns:
      A new Param instance.
    """
    current = {
      'value': self.value,
      'trainable': self.trainable,
      'metadata': self.metadata,
    }
    current.update(updates)
    return Param(**current)

  def tree_flatten(
    self,
  ) -> tuple[tuple[Any], tuple[bool, dict[str, Any]]]:
    """Flattens the param for JAX pytree registration."""
    return (self.value,), (self.trainable, self.metadata)

  @classmethod
  def tree_unflatten(
    cls,
    aux: tuple[bool, dict[str, Any]],
    children: tuple[Any],
  ) -> Param:
    """Unflattens the param for JAX pytree registration."""
    return cls(children[0], trainable=aux[0], metadata=aux[1])

  def __repr__(self) -> str:
    status = 'T' if self.trainable else 'N'
    parts = [f'value={self.value!r}']
    if self.metadata:
      parts.append(f'metadata={self.metadata}')
    return f'Param[{status}]({", ".join(parts)})'


jax.tree_util.register_pytree_node(
  Param, Param.tree_flatten, Param.tree_unflatten
)


class Params:
  """Immutable container for model parameters and state.

  This class manages the functional state of the model. It handles:
  1. Parameter storage and retrieval via tuple paths.
  2. Deterministic RNG key generation via an Rng module.
  3. Partitioning state for JIT compilation (trainable vs non-trainable).

  For sharded/vmapped code, use fold_in_axes() at the start of the function
  to get device-unique RNG keys while keeping the master key replicated.
  """

  def __init__(self, *, rng: 'Rng') -> None:
    """Initializes the container with an Rng module.

    Args:
      rng: An Rng module for random number generation.
    """
    self._data: dict[Path, Param] = {}
    self._initialized: bool = False
    # Tracks which axes have been folded in (ordered tuple for determinism).
    self._folded_axes: tuple[str, ...] = ()
    # Reference to the Rng module for key generation.
    self._rng: 'Rng' = rng

  @property
  def initialized(self) -> bool:
    """Returns True if the parameter initialization has been finalized."""
    return self._initialized

  @property
  def folded_axes(self) -> tuple[str, ...]:
    """Returns the currently folded axes for device-unique RNG."""
    return self._folded_axes

  def __repr__(self) -> str:
    n_vars = len(self._data)
    status = 'initialized' if self._initialized else 'uninitialized'
    return f'Params({n_vars} variables, {status})'

  def fold_in_axes(self, *axis_names: str) -> 'Params':
    """Records axes to fold in for device-unique RNG.

    Call this at the start of a shard_map or vmap function to get different
    RNG keys on each device/batch element. The Rng module reads folded_axes
    when generating keys.

    Folding is idempotent per axis - folding the same axis twice has no effect.
    Axes are folded in order, which affects the resulting key.

    Use fold_out_axes() to explicitly unfold axes before returning from shard_map.

    Args:
      *axis_names: Axis names to fold in (e.g., 'model', 'batch').

    Returns:
      Self if all axes already folded, otherwise a new Params with axes recorded.

    Raises:
      ValueError: If an axis doesn't exist in the current context.

    Example:
      @jax.shard_map(mesh=mesh, in_specs=P(), out_specs=param_specs)
      def init_sharded(x):
        params = bx.Params(rng=rng).fold_in_axes('model')
        _, params = model(params, x)
        return params.fold_out_axes('model').finalize()
    """
    if not axis_names:
      return self

    # Check if axes exist using JAX internal API.
    current_axes = jax.core.unsafe_get_axis_names_DO_NOT_USE()

    # If no axes are active (e.g., outside shard_map), silently skip.
    # This allows the same code to work both inside and outside shard_map.
    if not current_axes:
      return self

    # Validate all axes exist.
    for axis_name in axis_names:
      if axis_name not in current_axes:
        raise ValueError(
          f"Axis '{axis_name}' not found. Available axes: {current_axes}. "
          f'Make sure fold_in_axes is called inside shard_map/vmap with this axis.'
        )

    # Check if all axes are already folded (idempotent).
    new_axes = [name for name in axis_names if name not in self._folded_axes]
    if not new_axes:
      return self

    # Record folded axes (actual folding happens in Rng).
    new_p = self._clone()
    for axis_name in new_axes:
      new_p._folded_axes = new_p._folded_axes + (axis_name,)
    return new_p

  def fold_out_axes(self, *axis_names: str) -> 'Params':
    """Removes axes from the folded axes list.

    Call this before returning from shard_map/vmap to ensure pytree metadata
    matches between eval_shape (outside) and actual execution (inside).

    The axes to unfold must be at the end of the folded axes stack. The order
    of axes in the call doesn't matter, but they must all be at the tail.

    Args:
      *axis_names: Axis names to unfold. Must be at the end of _folded_axes.

    Returns:
      A new Params with the axes removed from folded_axes.

    Raises:
      ValueError: If axes are not at the end of the folded axes stack.

    Example:
      @jax.shard_map(mesh=mesh, in_specs=P(), out_specs=param_specs)
      def apply_sharded(params, x):
        params = params.fold_in_axes('model')
        out, params = model(params, x)
        return out, params.fold_out_axes('model')
    """
    if not axis_names:
      return self

    # If no axes are active (outside shard_map), silently skip.
    current_axes = jax.core.unsafe_get_axis_names_DO_NOT_USE()
    if not current_axes:
      return self

    # Validate axes are in folded_axes.
    axes_set = set(axis_names)
    for axis_name in axis_names:
      if axis_name not in self._folded_axes:
        raise ValueError(
          f"Axis '{axis_name}' is not folded. Current folded axes: "
          f'{self._folded_axes}'
        )

    # Validate axes form the tail of the stack.
    n_to_remove = len(axes_set)
    tail_axes = set(self._folded_axes[-n_to_remove:])
    if axes_set != tail_axes:
      raise ValueError(
        f'Axes {axis_names} must be at the end of folded axes stack. '
        f'Current stack: {self._folded_axes}. '
        f'Tail axes: {tail_axes}'
      )

    # Remove axes from the end.
    new_p = self._clone()
    new_p._folded_axes = self._folded_axes[:-n_to_remove]
    return new_p

  def next_key(self) -> tuple[jax.Array, Params]:
    """Generates a new key using the internal Rng module.

    Delegates to the Rng module which handles key generation, counter
    management, and folded axes for device-unique randomness.

    Returns:
      A tuple containing (new_key, new_params_container).

    Raises:
      ValueError: If no Rng was configured.
    """
    return self._rng(self)

  def get(
    self,
    graph: Graph,
    name: str,
    shape: Shape,
    init: Initializer,
    dtype: jnp.dtype = jnp.float32,
    trainable: bool = True,
    metadata: dict[str, Any] | None = None,
  ) -> tuple[jax.Array, Params]:
    """Retrieves an existing parameter or creates a new one.

    Args:
      graph: The graph node requesting the parameter.
      name: The local name of the parameter.
      shape: The shape of the parameter tensor.
      init: Function to initialize the parameter.
      dtype: The data type.
      trainable: Whether the parameter is trainable.
      metadata: Optional metadata dictionary. Common keys include:
        - 'sharding': tuple of mesh axis names (e.g., (None, 'model'))

    Returns:
      A tuple containing (parameter_value, new_params_container).

    Raises:
      KeyError: If parameters are finalized and the key is missing.
    """
    full_path = graph.path + (name,)

    # Check for existing parameter.
    if full_path in self._data:
      return self._data[full_path].value, self

    # Check against adding new params if finalized.
    if self._initialized:
      path_str = '/'.join(full_path)
      raise KeyError(f"Parameter '{path_str}' is missing.")

    # Try initializer with key=None first (for constant initializers).
    # If that fails, use a fresh key from RNG.
    try:
      val = init(None, shape, dtype)  # type: ignore
      new_p = self._clone()
    except Exception:
      # Initializer requires a key, so get one from RNG.
      key, new_p = self.next_key()
      val = init(key, shape, dtype)

    var = Param(val, trainable=trainable, metadata=metadata)

    # Store the variable.
    new_p._data[full_path] = var
    return val, new_p

  def set(self, path: Path, value: Any) -> Params:
    """Updates the value of an existing parameter.

    Args:
      path: The full path tuple of the parameter, e.g. ('net', 'linear', 'w').
      value: The new value (must match the dtype of the existing variable).

    Returns:
      A new Params container with the updated value.

    Raises:
      KeyError: If the path does not exist.
    """
    if path not in self._data:
      path_str = '/'.join(path)
      raise KeyError(f"Path '{path_str}' not found.")

    current_var = self._data[path]
    new_var = current_var.replace(value=value)

    new_p = self._clone()
    new_p._data[path] = new_var
    return new_p

  def split(
    self, predicate: Callable[[Path, Param], bool] | None = None
  ) -> tuple[Params, Params]:
    """Splits params into two containers based on a predicate.

    When called without arguments, splits by trainable flag
    (trainable vs non-trainable) otherwise splits by the provided predicate.

    Args:
      predicate: Optional function taking (param_path, param) and returning
        bool. param_path is a tuple like ('net', 'encoder', 'w'), where the last
        entry is the param name. If None, defaults to splitting into trainable
        and non-trainable params.

    Returns:
      A tuple (matching_params, non_matching_params).
    """

    def _default_predicate(_path: Path, p: Param) -> bool:
      return p.trainable

    if predicate is None:
      predicate = _default_predicate

    t_data: dict[Path, Param] = {}
    f_data: dict[Path, Param] = {}
    for full_path, param in self._data.items():
      if predicate(full_path, param):
        t_data[full_path] = param
      else:
        f_data[full_path] = param

    t, f = self._clone(), self._clone()
    t._data, f._data = t_data, f_data
    return t, f

  def merge(self, other: Params) -> Params:
    """Combines this container with another.

    Keys in 'other' override keys in 'self'.

    Args:
      other: The params container to merge in.

    Returns:
      A new merged Params container.
    """
    p = self._clone()
    p._data.update(other._data)
    return p

  def finalize(self) -> Params:
    """Marks initialization as complete, freezing the set of keys.

    Returns:
      A new Params container marked as initialized.
    """
    p = self._clone()
    p._initialized = True
    return p

  def _clone(self) -> Params:
    """Internal helper to clone the container."""
    # We must cast the raw object to Params so the type checker knows
    # it has the _data and _initialized attributes.
    p = cast(Params, object.__new__(Params))
    p._data = self._data.copy()
    p._initialized = self._initialized
    p._folded_axes = self._folded_axes
    p._rng = self._rng

    return p

  def tree_flatten(
    self,
  ) -> tuple[tuple[dict[Path, Param]], tuple[bool, tuple[str, ...], 'Rng']]:
    """Flattens the container for JAX pytree registration."""
    return (self._data,), (self._initialized, self._folded_axes, self._rng)

  @classmethod
  def tree_unflatten(
    cls,
    aux: tuple[bool, tuple[str, ...], 'Rng'],
    children: tuple[dict[Path, Param]],
  ) -> Params:
    """Unflattens the container for JAX pytree registration."""
    p = cast(Params, object.__new__(cls))
    p._data = children[0]
    p._initialized = aux[0]
    p._folded_axes = aux[1]
    p._rng = aux[2]
    return p


jax.tree_util.register_pytree_node(
  Params, Params.tree_flatten, Params.tree_unflatten
)


# ==============================================================================
# Base Modules
# ==============================================================================


class Module:
  """Base class for Neural Network layers.

  All layers should inherit from this class. It provides the connection to the
  Graph and helper methods for parameter creation.

  Note: This class does NOT enforce a specific `__call__` signature, as
  different modules require different inputs (e.g., Linear vs Attention).
  Subclasses should define `__call__` accepting `params` as the first argument.
  """

  def __init__(self, graph: Graph) -> None:
    """Initializes the Module with a graph node.

    Args:
      graph: The graph node representing this module's scope.
    """
    # Prevent binding directly to the root graph.
    if graph._is_root:
      raise ValueError(
        f"Cannot bind module '{self.__class__.__name__}' directly to the "
        f"root graph node '{graph.name}'. Please create a child scope "
        f"using `graph.child('name')`."
      )

    if '__type__' in graph.metadata:
      owner = graph.metadata['__type__']
      raise ValueError(
        f"Graph node '{graph.name}' is already owned by '{owner}'. "
        "Did you forget to call graph.child('name')?"
      )

    self.graph = graph
    self._capture_constructor_args()

  def _capture_constructor_args(self) -> None:
    """Captures the subclass constructor arguments into graph metadata."""
    # Register the class name.
    self.graph.metadata['__type__'] = self.__class__.__name__

    # Walk back up the stack to find the subclass __init__ frame.
    frame = inspect.currentframe().f_back.f_back  # type: ignore

    if frame:
      # Get explicit argument names and values.
      arg_info = inspect.getargvalues(frame)

      config = {}

      # Capture standard arguments (positional/keyword).
      for arg_name in arg_info.args:
        if arg_name not in ('self', 'graph', '__class__'):
          config[arg_name] = arg_info.locals[arg_name]
      if arg_info.keywords:
        kwargs = arg_info.locals[arg_info.keywords]
        config.update(kwargs)

      # Filter out private attributes.
      clean_config = {k: v for k, v in config.items() if not k.startswith('_')}

      self.graph.metadata.update(clean_config)

  def get_param(
    self,
    params: Params,
    name: str,
    shape: Shape,
    init: Initializer,
    dtype: jnp.dtype = jnp.float32,
    trainable: bool = True,
    metadata: dict[str, Any] | None = None,
  ) -> tuple[jax.Array, Params]:
    """Shortcut to create parameters within this module's graph scope.

    Args:
      params: The parameters container.
      name: The local name of the parameter (appended to graph path).
      shape: The shape of the parameter.
      init: The initialization function.
      dtype: The data type.
      trainable: Whether the parameter is trainable.
      metadata: Optional metadata dictionary. Common keys include:
        - 'sharding': tuple of mesh axis names (e.g., (None, 'model'))

    Returns:
      A tuple containing (parameter_value, new_params_container).
    """
    return params.get(self.graph, name, shape, init, dtype, trainable, metadata)

  def set_param(self, params: Params, name: str, value: Any) -> Params:
    """Shortcut to update a parameter within this module's graph scope.

    Args:
      params: The parameters container.
      name: The local name of the parameter (appended to graph path).
      value: The new value.

    Returns:
      A new Params container with the updated value.
    """
    full_path = self.graph.path + (name,)
    return params.set(full_path, value)


class Rng(Module):
  """A random number generator stream stored as non-trainable params.

  Produces deterministic, counter-based random keys. Reads folded_axes
  from Params (via public property) for device-unique randomness in vmap/shard_map.

  Uses get_param like any other module - no internal state access.

  Example:
    # Create Params with an Rng
    graph = bx.Graph('root')
    params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))

    # Get a key
    key, params = params.next_key()

    # Module-owned RNG for Dropout
    class MyModel(bx.Module):
      def __init__(self, graph):
        super().__init__(graph)
        self.dropout_rng = bx.Rng(graph.child('dropout_rng'), seed=0)
        self.dropout = bx.Dropout(
          graph.child('dropout'), rate=0.5, rng=self.dropout_rng
        )

      def __call__(self, params, x, is_training=True):
        x, params = self.dropout(params, x, is_training)
        return x, params
  """

  def __init__(self, graph: Graph, seed: int | jax.Array) -> None:
    """Initializes the Rng module.

    Args:
      graph: The graph node for this module's scope.
      seed: Integer seed or JAX key array.
    """
    super().__init__(graph)
    self.seed = seed
    # Store initial key value (will be initialized via get_param on first call).
    if isinstance(seed, int):
      self._init_key = jax.random.key(seed)
    else:
      self._init_key = seed

  def __call__(self, params: Params) -> tuple[jax.Array, Params]:
    """Generate next key, respecting params' folded axes.

    The key and counter are stored as non-trainable params under this
    module's graph path (using get_param). Each call increments the counter.

    Args:
      params: The params container.

    Returns:
      Tuple of (new_key, updated_params).
    """
    # Get or create key/counter using constant initializers (no RNG needed).
    key_init = jax.nn.initializers.constant(
      self._init_key, self._init_key.dtype
    )
    counter_init = jax.nn.initializers.constant(0, dtype=jnp.uint32)

    base_key, params = self.get_param(
      params,
      'key',
      self._init_key.shape,
      key_init,
      dtype=self._init_key.dtype,
      trainable=False,
    )
    counter, params = self.get_param(
      params, 'counter', (), counter_init, dtype=jnp.uint32, trainable=False
    )

    # Fold in any axes from params.folded_axes (public property).
    folded_key = base_key
    for axis_name in params.folded_axes:
      axis_idx = jax.lax.axis_index(axis_name)
      folded_key = jax.random.fold_in(folded_key, axis_idx)

    # Fold in counter for deterministic sequence.
    new_key = jax.random.fold_in(folded_key, counter)

    # Increment counter using set_param.
    params = self.set_param(params, 'counter', counter + 1)

    return new_key, params


# ==============================================================================
# Sequence Processing & Scanning Logic
# ==============================================================================

StepFn = Callable[
  [Params, InputsT, StateT, ResetT | None, bool],
  tuple[tuple[OutputsT, StateT], Params],
]


def _swap_batch_time(x: jax.Array) -> jax.Array:
  """Swaps axis 0 and 1 of the input array."""
  return jnp.swapaxes(x, 0, 1)


def _scan_init(
  step_fn: StepFn[InputsT, StateT, OutputsT, ResetT],
  params: Params,
  inputs: InputsT,
  prev_state: StateT,
  is_reset: ResetT | None,
  is_training: bool,
) -> tuple[tuple[OutputsT, StateT], Params]:
  """Performs a single initialization step and expands output."""
  # Slice inputs to time 0.
  inputs_t0 = jax.tree.map(lambda x: x[:, 0], inputs)
  reset_t0 = jax.tree.map(lambda x: jnp.ones_like(x[:, 0]), is_reset)

  # Run one step to initialize parameters.
  (out_t0, new_state), new_params = step_fn(
    params, inputs_t0, prev_state, reset_t0, is_training
  )

  # Get sequence length.
  T = jax.tree.leaves(inputs)[0].shape[1]
  outputs = jax.tree.map(lambda x: jnp.stack([x] * T, axis=1), out_t0)

  return (outputs, new_state), new_params


def static_scan(
  step_fn: StepFn[InputsT, StateT, OutputsT, ResetT],
  params: Params,
  inputs: InputsT,
  prev_state: StateT,
  is_reset: ResetT | None,
  is_training: bool,
) -> tuple[tuple[OutputsT, StateT], Params]:
  """Performs a Python loop scan over the time dimension.

  This function explicitly iterates over the time dimension (axis 1) of the
  inputs using a Python `for` loop. This is useful for debugging, handling
  control flow that `jax.lax.scan` cannot compile, or when the sequence length
  is very short.

  Args:
    step_fn: A callable that processes a single time step. Signature:
      (params, inputs, state, is_reset, is_training) -> ((output, state), params)
      This can be a method like `model.__call__` or a partial function.
    params: The parameters container.
    inputs: Input sequence Pytree [Batch, Time, ...].
    prev_state: Initial state.
    is_reset: Optional reset signal [Batch, Time].
    is_training: Training flag.

  Returns:
    ((outputs, final_state), updated_params)

  Raises:
    ValueError: If inputs are empty or have invalid rank.
  """
  leaves = jax.tree.leaves(inputs)
  if not leaves:
    raise ValueError('The input Pytree cannot be empty.')

  for x in leaves:
    if x.ndim < 2:
      raise ValueError(f'Input leaves must have rank >= 2, got {x.ndim}.')

  # Verify all inputs have the same time dimension.
  T = leaves[0].shape[1]
  for x in leaves:
    chex.assert_axis_dimension(x, axis=1, expected=T)

  if not params.initialized:
    return _scan_init(
      step_fn, params, inputs, prev_state, is_reset, is_training
    )

  outputs_list = []
  current_state = prev_state
  current_params = params

  for t in range(T):
    inputs_t = jax.tree.map(lambda x: x[:, t], inputs)
    reset_t = jax.tree.map(lambda x: x[:, t], is_reset)

    # Returns ((out, state), params).
    (out_t, current_state), current_params = step_fn(
      current_params, inputs_t, current_state, reset_t, is_training
    )
    outputs_list.append(out_t)

  outputs = jax.tree.map(lambda *args: jnp.stack(args, axis=1), *outputs_list)
  return (outputs, current_state), current_params


def dynamic_scan(
  step_fn: StepFn[InputsT, StateT, OutputsT, ResetT],
  params: Params,
  inputs: InputsT,
  prev_state: StateT,
  is_reset: ResetT | None,
  is_training: bool,
) -> tuple[tuple[OutputsT, StateT], Params]:
  """Performs a compiled jax.lax.scan over the time dimension.

  This uses XLA compilation for high performance on long sequences.

  Args:
    step_fn: A callable that processes a single time step. Signature:
      (params, inputs, state, is_reset, is_training) -> ((output, state), params)
      This can be a method like `model.__call__` or a partial function.
    params: The parameters container.
    inputs: Input sequence Pytree [Batch, Time, ...].
    prev_state: Initial state.
    is_reset: Optional reset signal [Batch, Time].
    is_training: Training flag.

  Returns:
    ((outputs, final_state), updated_params)

  Raises:
    ValueError: If inputs have invalid rank.
  """
  leaves = jax.tree.leaves(inputs)
  for x in leaves:
    if x.ndim < 2:
      raise ValueError(f'Input leaves must have rank >= 2, got {x.ndim}.')

  # Verify all inputs have the same time dimension.
  T = leaves[0].shape[1]
  for x in leaves:
    chex.assert_axis_dimension(x, axis=1, expected=T)

  if not params.initialized:
    return _scan_init(
      step_fn, params, inputs, prev_state, is_reset, is_training
    )

  # Swap to [Time, Batch, ...]
  inputs_t = jax.tree.map(_swap_batch_time, inputs)
  reset_scan = jax.tree.map(_swap_batch_time, is_reset)

  def scan_body(carry: Any, scan_inputs: Any) -> tuple[Any, Any]:
    curr_state, curr_params = carry
    inputs_step, reset_step = scan_inputs

    (out, next_state), next_params = step_fn(
      curr_params, inputs_step, curr_state, reset_step, is_training
    )
    # scan expects ((next_carry), output)
    return (next_state, next_params), out

  (final_state, final_params), outputs_t = jax.lax.scan(
    scan_body, (prev_state, params), (inputs_t, reset_scan)
  )

  outputs = jax.tree.map(_swap_batch_time, outputs_t)
  return (outputs, final_state), final_params


class SequenceBase(Module, Generic[InputsT, StateT, OutputsT, ResetT]):
  """Base class for sequence-processing modules.

  This abstract class defines the interface for modules that process sequences.
  It supports both 'chunk' processing (e.g., Transformers) and 'step' processing
  (e.g., RNNs). Unlike the base Module, SequenceBase enforces a specific
  call signature.

  The primary method is `__call__` for single-step processing. For sequence
  processing, use `apply` which internally uses `static_scan` or `dynamic_scan`.
  """

  def initial_state(
    self, params: Params, inputs: InputsT
  ) -> tuple[StateT, Params]:
    """Computes the initial state for the sequence processing.

    Args:
      params: The parameters container.
      inputs: The input Pytree. Used to infer batch size or other
        structural properties.

    Returns:
      A tuple containing the initial state and the parameters container.
    """
    raise NotImplementedError

  def __call__(
    self,
    params: Params,
    inputs: InputsT,
    prev_state: StateT | None,
    is_reset: ResetT | None = None,
    is_training: bool = True,
  ) -> tuple[tuple[OutputsT, StateT], Params]:
    """Processes a single time step of data.

    This is the primary method that subclasses must implement.

    Args:
      params: The parameters container.
      inputs: The input step Pytree. Leaves should have shape [Batch, ...].
      prev_state: The previous state.
      is_reset: Optional reset signal. Leaves should have shape [Batch].
      is_training: Boolean flag indicating if the model is in training mode.

    Returns:
      A nested tuple ((output, new_state), updated_params).
    """
    raise NotImplementedError

  def apply(
    self,
    params: Params,
    inputs: InputsT,
    prev_state: StateT | None = None,
    is_reset: ResetT | None = None,
    is_training: bool = True,
  ) -> tuple[tuple[OutputsT, StateT], Params]:
    """Processes a sequence of data [Batch, Time, ...].

    Default behavior: Wraps __call__ by iterating over the time dimension.
    Subclasses may override this for more efficient sequence processing.

    Args:
      params: The parameters container.
      inputs: The input sequence Pytree. Leaves should have shape
        [Batch, Time, ...].
      prev_state: Optional initial state. If None, `initial_state` is called.
      is_reset: Optional reset signal Pytree. Leaves should have shape
        [Batch, Time].
      is_training: Boolean flag indicating if the model is in training mode.

    Returns:
      A nested tuple ((outputs, final_state), updated_params).
    """
    raise NotImplementedError


class RecurrenceBase(SequenceBase[InputsT, StateT, OutputsT, ResetT]):
  """Base class for Recurrent Neural Networks (RNNs).

  Implements sequence processing by scanning over the `__call__` method.
  Handles automatic fallback to static unrolling during initialization.

  Subclasses must implement:
  - `initial_state`: Returns the initial hidden state.
  - `__call__`: Processes a single time step.
  """

  def __init__(self, graph: Graph, is_static: bool = False) -> None:
    """Initializes the RecurrenceBase.

    Args:
      graph: The graph node for this module.
      is_static: If True, forces the use of Python loops (`static_scan`).
        If False, uses `dynamic_scan` (jax.lax.scan) for better performance.
    """
    super().__init__(graph)
    self._is_static = is_static

  @property
  def is_static(self) -> bool:
    """Returns whether the module is configured to use static unrolling."""
    return self._is_static

  @is_static.setter
  def is_static(self, value: bool) -> None:
    """Sets the unrolling strategy."""
    self._is_static = value

  def maybe_reset_state(
    self,
    params: Params,
    prev_state: StateT,
    inputs: InputsT,
    is_reset: ResetT | None = None,
  ) -> StateT:
    """Helper to reset state based on boolean signal.

    Args:
      params: The parameters container.
      prev_state: The current state Pytree.
      inputs: The current input step. Used to infer batch size for fresh state.
      is_reset: A boolean Pytree indicating which batch elements to reset.

    Returns:
      The updated state with resets applied where indicated.
    """
    if is_reset is None:
      return prev_state

    # Generate a fresh initial state for this batch.
    initial_state, _ = self.initial_state(params, inputs)

    if isinstance(is_reset, jax.Array):
      state = jax.tree.map(
        lambda i, p, r=is_reset: jnp.where(r, i, p), initial_state, prev_state
      )
    else:
      state = jax.tree.map(
        lambda i, p, r: jnp.where(r, i, p), initial_state, prev_state, is_reset
      )
    return cast(StateT, state)

  def __call__(
    self,
    params: Params,
    inputs: InputsT,
    prev_state: StateT | None,
    is_reset: ResetT | None = None,
    is_training: bool = True,
  ) -> tuple[tuple[OutputsT, StateT], Params]:
    """Processes a single time step of data.

    This method must be implemented by subclasses.

    Args:
      params: The parameters container.
      inputs: The input step Pytree. Leaves must have shape [Batch, ...].
      prev_state: The previous recurrent state. Cannot be None.
      is_reset: Optional reset signal. Leaves must have shape [Batch].
      is_training: Boolean flag indicating if the model is in training mode.

    Returns:
      A nested tuple ((output, new_state), updated_params).
    """
    raise NotImplementedError

  def apply(
    self,
    params: Params,
    inputs: InputsT,
    prev_state: StateT | None = None,
    is_reset: ResetT | None = None,
    is_training: bool = True,
  ) -> tuple[tuple[OutputsT, StateT], Params]:
    """Processes a sequence by scanning over __call__.

    This method automatically handles initialization: if parameters are not
    yet initialized, it forces a single-step execution expanded to the full
    sequence length to safely create parameters without violating JAX scan
    invariants.

    Args:
      params: The parameters container.
      inputs: The input sequence Pytree. Leaves must have shape
        [Batch, Time, ...].
      prev_state: Optional initial state. If None, `initial_state` is called.
      is_reset: Optional reset signal Pytree. Leaves must have shape
        [Batch, Time].
      is_training: Boolean flag indicating if the model is in training mode.

    Returns:
      A nested tuple ((outputs, final_state), updated_params).

    Raises:
      ValueError: If inputs have rank < 2.
    """
    if prev_state is None:
      prev_state, params = self.initial_state(params, inputs)

    for x in jax.tree.leaves(inputs):
      if x.ndim < 2:
        raise ValueError('Input leaves must have rank >= 2.')

    # Cast self to help type inference with generic parameters.
    step_fn = cast(StepFn[InputsT, StateT, OutputsT, ResetT], self)
    if self.is_static:
      return static_scan(
        step_fn, params, inputs, prev_state, is_reset, is_training
      )
    else:
      return dynamic_scan(
        step_fn, params, inputs, prev_state, is_reset, is_training
      )
