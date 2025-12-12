"""
Core interfaces and abstractions for the blox library.

This module defines the fundamental building blocks of the library, including:
- Graph: Structural definition of the model hierarchy.
- Params: Immutable, functional state container.
- Module: Base class for neural network layers.
- Sequential/RNNCore: Interfaces for time-series processing.

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

InputT = TypeVar('InputT', bound=chex.ArrayTree)
StateT = TypeVar('StateT', bound=chex.ArrayTree)
OutputT = TypeVar('OutputT', bound=chex.ArrayTree)
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
  def sharding(self) -> tuple[str | None, ...] | None:
    """Returns the sharding spec from metadata, if present."""
    return self.metadata.get('sharding')

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
  """Immutable container for variables and RNG state.

  This class manages the functional state of the model. It handles:
  1. Parameter storage and retrieval via tuple paths.
  2. Deterministic RNG key splitting.
  3. Partitioning state for JIT compilation (trainable vs non-trainable).

  For sharded/vmapped code, use fold_in_axes() at the start of the function
  to get device-unique RNG keys while keeping the master key replicated.
  """

  def __init__(self, *, seed: int | jax.Array) -> None:
    """Initializes the container with a master random seed.

    Args:
      seed: An integer seed or a JAX random key.
    """
    self._data: dict[Path, Param] = {}
    self._initialized: bool = False
    # Tracks which axes have been folded in (ordered tuple for determinism).
    # Part of pytree metadata, restored on unflatten.
    self._folded_axes: tuple[str, ...] = ()
    # Local key derived from master key + folded axes. None until fold_in_axes.
    self._folded_key: jax.Array | None = None

    if isinstance(seed, int):
      key = jax.random.key(seed)
    else:
      key = seed

    # Store RNG key and counter as separate Params for clean pytree structure.
    self._data[('rng', 'key')] = Param(key, trainable=False)
    self._data[('rng', 'counter')] = Param(
      jnp.array(0, dtype=jnp.uint32), trainable=False
    )

  @property
  def initialized(self) -> bool:
    """Returns True if the parameter initialization has been finalized."""
    return self._initialized

  def __repr__(self) -> str:
    n_vars = len(self._data)
    status = 'initialized' if self._initialized else 'uninitialized'
    return f'Params({n_vars} variables, {status})'

  def fold_in_axes(self, *axis_names: str) -> 'Params':
    """Folds axis indices into the RNG for device-unique randomness.

    Call this at the start of a shard_map or vmap function to get different
    RNG keys on each device/batch element. The master key stays replicated,
    but subsequent next_key() calls will produce device-unique keys.

    Folding is idempotent per axis - folding the same axis twice has no effect.
    Axes are folded in order, which affects the resulting key.

    Use fold_out_axes() to explicitly unfold axes before returning from shard_map.

    Args:
      *axis_names: Axis names to fold in (e.g., 'model', 'batch').

    Returns:
      A new Params with the axes folded in.

    Raises:
      ValueError: If an axis doesn't exist in the current context.

    Example:
      @jax.shard_map(mesh=mesh, in_specs=P(), out_specs=param_specs)
      def init_sharded(x):
        params = bx.Params(seed=42).fold_in_axes('model')
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

    # Fold in each axis sequentially.
    result = self
    for axis_name in axis_names:
      if axis_name in result._folded_axes:
        continue  # Already folded, skip.

      # Compute new folded key by folding in the axis index.
      base_key = (
        result._folded_key
        if result._folded_key is not None
        else result._data[('rng', 'key')].value
      )
      axis_idx = jax.lax.axis_index(axis_name)
      new_folded_key = jax.random.fold_in(base_key, axis_idx)

      new_p = result._clone()
      new_p._folded_axes = result._folded_axes + (axis_name,)
      new_p._folded_key = new_folded_key
      result = new_p

    return result

  def fold_out_axes(self, *axis_names: str) -> 'Params':
    """Unfolds axis indices from the RNG, reverting to outer context.

    Call this before returning from shard_map/vmap to ensure pytree metadata
    matches between eval_shape (outside) and actual execution (inside).

    The axes to unfold must be at the end of the folded axes stack. The order
    of axes in the call doesn't matter, but they must all be at the tail.

    Args:
      *axis_names: Axis names to unfold. Must be at the end of _folded_axes.

    Returns:
      A new Params with the axes unfolded.

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
    new_folded_axes = self._folded_axes[:-n_to_remove]

    # Recompute folded_key from remaining axes.
    new_p = self._clone()
    new_p._folded_axes = ()
    new_p._folded_key = None

    # Re-fold remaining axes.
    for axis_name in new_folded_axes:
      base_key = (
        new_p._folded_key
        if new_p._folded_key is not None
        else new_p._data[('rng', 'key')].value
      )
      axis_idx = jax.lax.axis_index(axis_name)
      new_p._folded_key = jax.random.fold_in(base_key, axis_idx)
      new_p._folded_axes = new_p._folded_axes + (axis_name,)

    return new_p

  def next_key(self) -> tuple[jax.Array, Params]:
    """Generates a new key and increments the internal counter.

    This ensures functional purity. The method returns a new key and a *new*
    Params container with the updated counter.

    If fold_in_axes() was called, the generated key will be derived from
    the folded key, producing device-unique keys inside shard_map/vmap.

    Returns:
      A tuple containing (new_key, new_params_container).

    Raises:
      ValueError: If the 'rng' state is missing.
    """
    key_path = ('rng', 'key')
    counter_path = ('rng', 'counter')

    if key_path not in self._data or counter_path not in self._data:
      raise ValueError('Params container must contain RNG state.')

    # Use folded key if available, otherwise master key.
    base_key = (
      self._folded_key
      if self._folded_key is not None
      else self._data[key_path].value
    )
    counter = self._data[counter_path].value

    # Generate deterministic key based on current counter.
    new_key = jax.random.fold_in(base_key, counter)

    # Increment counter.
    new_counter = counter + 1

    # Return updated container.
    new_p = self._clone()
    new_p._data[counter_path] = self._data[counter_path].replace(
      value=new_counter
    )
    return new_key, new_p

  def get(
    self,
    graph: Graph,
    name: str,
    shape: Shape,
    initializer: Initializer,
    dtype: Any = float,
    trainable: bool = True,
    metadata: dict[str, Any] | None = None,
  ) -> tuple[jax.Array, Params]:
    """Retrieves an existing parameter or creates a new one.

    Args:
      graph: The graph node requesting the parameter.
      name: The local name of the parameter.
      shape: The shape of the parameter tensor.
      initializer: Function to initialize the parameter.
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

    # Consume RNG to generate a fresh key.
    key, new_p = self.next_key()

    # Initialize the new value.
    val = initializer(key, shape, dtype)
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
    val_arr = jnp.array(value, dtype=current_var.value.dtype)
    new_var = current_var.replace(value=val_arr)

    new_p = self._clone()
    new_p._data[path] = new_var
    return new_p

  def split(
    self, predicate: Callable[[Path, str, Param], bool] | None = None
  ) -> tuple[Params, Params]:
    """Splits params into two containers based on a predicate.

    When called without arguments, splits by trainable flag
    (trainable vs non-trainable) otherwise splits by the provided predicate.

    Args:
      predicate: Optional function taking (module_path, param_name, param)
        and returning bool. module_path is a tuple like ('net', 'encoder'),
        param_name is the variable name like 'w'. If None, defaults to
        splitting by trainable flag.

    Returns:
      A tuple (matching_params, non_matching_params).
    """

    def _default_predicate(_path: Path, _name: str, p: Param) -> bool:
      return p.trainable

    if predicate is None:
      predicate = _default_predicate

    t_data: dict[Path, Param] = {}
    f_data: dict[Path, Param] = {}
    for full_path, param in self._data.items():
      # Split full_path tuple into module_path and param_name.
      module_path = full_path[:-1]
      param_name = full_path[-1]

      if predicate(module_path, param_name, param):
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
    p._folded_key = self._folded_key
    return p

  def tree_flatten(
    self,
  ) -> tuple[tuple[dict[Path, Param]], tuple[bool, tuple[str, ...]]]:
    """Flattens the container for JAX pytree registration.

    Note: folded_key is NOT included as a child. It will be reconstructed
    on unflatten from the folded_axes metadata.
    """
    return (self._data,), (self._initialized, self._folded_axes)

  @classmethod
  def tree_unflatten(
    cls,
    aux: tuple[bool, tuple[str, ...]],
    children: tuple[dict[Path, Param]],
  ) -> Params:
    """Unflattens the container for JAX pytree registration.

    Restores the params state as-is. Users are responsible for explicitly
    calling fold_out_axes() before returning from shard_map/vmap.
    """
    p = cast(Params, object.__new__(cls))
    p._data = children[0]
    p._initialized = aux[0]
    p._folded_axes = aux[1]

    # Reconstruct folded_key if there are folded axes.
    if p._folded_axes:
      current_axes = jax.core.unsafe_get_axis_names_DO_NOT_USE()
      if current_axes:
        # Re-fold each axis to reconstruct folded_key.
        base_key = p._data[('rng', 'key')].value
        for axis_name in p._folded_axes:
          axis_idx = jax.lax.axis_index(axis_name)
          base_key = jax.random.fold_in(base_key, axis_idx)
        p._folded_key = base_key
      else:
        # Outside transformation, can't reconstruct folded_key.
        p._folded_key = None
    else:
      p._folded_key = None

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
    initializer: Initializer,
    dtype: Any = float,
    trainable: bool = True,
    metadata: dict[str, Any] | None = None,
  ) -> tuple[jax.Array, Params]:
    """Shortcut to create parameters within this module's graph scope.

    Args:
      params: The parameters container.
      name: The local name of the parameter (appended to graph path).
      shape: The shape of the parameter.
      initializer: The initialization function.
      dtype: The data type.
      trainable: Whether the parameter is trainable.
      metadata: Optional metadata dictionary. Common keys include:
        - 'sharding': tuple of mesh axis names (e.g., (None, 'model'))

    Returns:
      A tuple containing (parameter_value, new_params_container).
    """
    return params.get(
      self.graph, name, shape, initializer, dtype, trainable, metadata
    )


# ==============================================================================
# Sequential & Scanning Logic
# ==============================================================================


def _swap_batch_time(x: jax.Array) -> jax.Array:
  """Swaps axis 0 and 1 of the input array."""
  return jnp.swapaxes(x, 0, 1)


def _scan_init(
  core: RNNCore[InputT, StateT, OutputT, ResetT],
  params: Params,
  inputs: InputT,
  prev_state: StateT,
  is_reset: ResetT | None,
  is_training: bool,
) -> tuple[tuple[OutputT, StateT], Params]:
  """Performs a single initialization step and expands output."""
  # Slice inputs to time 0.
  inputs_t0 = jax.tree.map(lambda x: x[:, 0], inputs)
  reset_t0 = jax.tree.map(lambda x: jnp.ones_like(x[:, 0]), is_reset)

  # Run one step to initialize parameters.
  (out_t0, new_state), new_params = core.step(
    params, inputs_t0, prev_state, reset_t0, is_training
  )

  # Get sequence length.
  T = jax.tree.leaves(inputs)[0].shape[1]
  outputs = jax.tree.map(lambda x: jnp.stack([x] * T, axis=1), out_t0)

  return (outputs, new_state), new_params


def static_scan(
  core: RNNCore[InputT, StateT, OutputT, ResetT],
  params: Params,
  inputs: InputT,
  prev_state: StateT,
  is_reset: ResetT | None,
  is_training: bool,
) -> tuple[tuple[OutputT, StateT], Params]:
  """Performs a Python loop scan over the time dimension.

  This function explicitly iterates over the time dimension (axis 1) of the
  inputs using a Python `for` loop. This is useful for debugging, handling
  control flow that `jax.lax.scan` cannot compile, or when the sequence length
  is very short.

  Args:
    core: The RNN module implementing the `step` method.
    params: The parameters container.
    inputs: Input sequence Pytree [Batch, Time, ...].
    prev_state: Initial state (must be initialized).
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

  if not params.initialized:
    return _scan_init(core, params, inputs, prev_state, is_reset, is_training)

  T = leaves[0].shape[1]

  outputs_list = []
  current_state = prev_state
  current_params = params

  for t in range(T):
    inputs_t = jax.tree.map(lambda x: x[:, t], inputs)
    reset_t = jax.tree.map(lambda x: x[:, t], is_reset)

    # Returns ((out, state), params)
    (out_t, current_state), current_params = core.step(
      current_params, inputs_t, current_state, reset_t, is_training
    )
    outputs_list.append(out_t)

  outputs = jax.tree.map(lambda *args: jnp.stack(args, axis=1), *outputs_list)
  return (outputs, current_state), current_params


def dynamic_scan(
  core: RNNCore[InputT, StateT, OutputT, ResetT],
  params: Params,
  inputs: InputT,
  prev_state: StateT,
  is_reset: ResetT | None,
  is_training: bool,
) -> tuple[tuple[OutputT, StateT], Params]:
  """Performs a compiled jax.lax.scan over the time dimension.

  This uses XLA compilation for high performance on long sequences.

  Args:
    core: The RNN module implementing the `step` method.
    params: The parameters container.
    inputs: Input sequence Pytree [Batch, Time, ...].
    prev_state: Initial state (must be initialized).
    is_reset: Optional reset signal [Batch, Time].
    is_training: Training flag.

  Returns:
    ((outputs, final_state), updated_params)

  Raises:
    ValueError: If inputs have invalid rank.
  """
  for x in jax.tree.leaves(inputs):
    if x.ndim < 2:
      raise ValueError(f'Input leaves must have rank >= 2, got {x.ndim}.')

  if not params.initialized:
    return _scan_init(core, params, inputs, prev_state, is_reset, is_training)

  # Swap to [Time, Batch, ...]
  inputs_t = jax.tree.map(_swap_batch_time, inputs)
  reset_scan = jax.tree.map(_swap_batch_time, is_reset)

  def scan_fn(carry: Any, scan_inputs: Any) -> tuple[Any, Any]:
    curr_state, curr_params = carry
    inputs_step, reset_step = scan_inputs

    (out, next_state), next_params = core.step(
      curr_params, inputs_step, curr_state, reset_step, is_training
    )
    # scan expects ((next_carry), output)
    return (next_state, next_params), out

  (final_state, final_params), outputs_t = jax.lax.scan(
    scan_fn, (prev_state, params), (inputs_t, reset_scan)
  )

  outputs = jax.tree.map(_swap_batch_time, outputs_t)
  return (outputs, final_state), final_params


class Sequential(Module, Generic[InputT, StateT, OutputT, ResetT]):
  """Interface for sequential time-series modules.

  This abstract class allows modules to define operations on sequences.
  It supports both 'chunk' processing (Transformers) and 'step' processing (RNNs).
  Unlike the base Module, Sequential enforces a specific call signature.
  """

  def initial_state(
    self, params: Params, inputs: InputT
  ) -> tuple[StateT, Params]:
    """Computes the initial state for the sequence processing.

    Args:
      params: The parameters container.
      inputs: The input sequence Pytree. Used to infer batch size or other
        structural properties.

    Returns:
      A tuple containing the initial state and the parameters container.
    """
    raise NotImplementedError

  def step(
    self,
    params: Params,
    inputs: InputT,
    prev_state: StateT | None,
    is_reset: ResetT | None = None,
    is_training: bool = True,
  ) -> tuple[tuple[OutputT, StateT], Params]:
    """Processes a single time step of data.

    Default behavior: Wraps __call__ by adding a fake time dimension.
    This allows Sequence models (like Attention) to work step-by-step.

    Args:
      params: The parameters container.
      inputs: The input step Pytree. Leaves should have shape [Batch, ...].
      prev_state: The previous recurrent state.
      is_reset: Optional reset signal. Leaves should have shape [Batch].
      is_training: Boolean flag indicating if the model is in training mode.

    Returns:
      A nested tuple ((output, new_state), updated_params).

    Raises:
      ValueError: If inputs have rank < 1.
    """
    for x in jax.tree.leaves(inputs):
      if x.ndim < 1:
        raise ValueError('Input leaves must have at least rank 1 (Batch).')

    # Add time dim: [B, ...] -> [B, 1, ...]
    inputs_seq = jax.tree.map(lambda x: x[:, None], inputs)
    is_reset_seq = jax.tree.map(lambda x: x[:, None], is_reset)

    ((out_seq, new_state), new_params) = self.__call__(
      params, inputs_seq, prev_state, is_reset_seq, is_training
    )

    # Remove time dim: [B, 1, ...] -> [B, ...]
    out = jax.tree.map(lambda x: x.squeeze(axis=1), out_seq)

    return (out, new_state), new_params

  def __call__(
    self,
    params: Params,
    inputs: InputT,
    prev_state: StateT | None = None,
    is_reset: ResetT | None = None,
    is_training: bool = True,
  ) -> tuple[tuple[OutputT, StateT], Params]:
    """Processes a sequence of data [Batch, Time, ...].

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


class RNNCore(Sequential[InputT, StateT, OutputT, ResetT]):
  """Base class for Recurrent Neural Networks (RNNs).

  Implements the sequence loop by scanning over the `step` method.
  Handles automatic fallback to static unrolling during initialization.
  """

  def __init__(self, graph: Graph, is_static: bool = False) -> None:
    """Initializes the RNNCore.

    Args:
      graph: The graph node for this module.
      is_static: If True, forces the use of Python loops (`static_scan`).
        If False, attempts to use `dynamic_scan` (jax.lax.scan).
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
    inputs: InputT,
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

  def step(
    self,
    params: Params,
    inputs: InputT,
    prev_state: StateT | None,
    is_reset: ResetT | None = None,
    is_training: bool = True,
  ) -> tuple[tuple[OutputT, StateT], Params]:
    """Computes the output and new state for a single time step.

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

  def __call__(
    self,
    params: Params,
    inputs: InputT,
    prev_state: StateT | None = None,
    is_reset: ResetT | None = None,
    is_training: bool = True,
  ) -> tuple[tuple[OutputT, StateT], Params]:
    """Orchestrates the scan loop.

    This method includes an automatic optimization: if the parameters are not
    yet initialized, it forces a single-step execution expanded to the full
    sequence length. This ensures parameters are created safely without
    violating JAX scan invariants.

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

    if self.is_static:
      return static_scan(
        self, params, inputs, prev_state, is_reset, is_training
      )
    else:
      return dynamic_scan(
        self, params, inputs, prev_state, is_reset, is_training
      )
