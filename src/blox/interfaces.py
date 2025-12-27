"""Core interfaces and abstractions for the blox library.

This module defines the fundamental building blocks:

1.  **Graph:** Represents the static, hierarchical structure of the model. It
    handles naming and pathing (e.g. `net/layer1/weights`) but stores no state.
2.  **Params:** A functional, immutable container for all model state (weights,
    RNG keys, batch norms stats). It is passed through every layer.
3.  **Module:** The base class for layers. It connects a `Graph` node to
    parameter creation logic (`get_param` / `set_param`).
4.  **Sequence Processing:**
    *   `SequenceBase`: Abstract interface for layers that handle sequences
        (RNN, Transformer).
    *   `RecurrenceBase`: Abstract interface for layers that process sequences
        step-by-step (RNN, LSTM, GRU).

These interfaces enforce functional purity and explicit state management,
making the library robust for JAX transformations.
"""

from __future__ import annotations

import abc
import functools
import inspect
from collections.abc import ItemsView, KeysView, ValuesView
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

  Params is a pure state store holding all model state: trainable weights,
  non-trainable values (like batch norm statistics), and RNG state. It enforces
  functional purity by returning new instances on every modification.

  Key features:
  - **Functional updates**: All methods return new Params instances.
  - **Tuple paths**: Parameters are keyed by tuples like ('net', 'linear', 'w').
  - **Trainable split**: Use `split()` to separate trainable from non-trainable.

  Example:
    graph = bx.Graph('net')
    rng = bx.Rng(graph.child('rng'))
    model = MyModel(graph.child('model'), rng=rng)

    # Create params and seed the Rng.
    params = rng.seed(bx.Params(), seed=42)

    # Forward pass creates parameters.
    _, params = model(params, x)
    params = params.finalized()

    # Training loop.
    trainable, non_trainable = params.split()
    grads = jax.grad(loss_fn)(trainable, non_trainable, x)
    trainable = jax.tree.map(lambda w, g: w - lr * g, trainable, grads)
    params = trainable.merge(non_trainable)
  """

  # ============================================================================
  # Initialization
  # ============================================================================

  def __init__(self) -> None:
    """Creates an empty parameter container."""
    self._data: dict[Path, Param] = {}
    self._initialized: bool = False

  # ============================================================================
  # Properties
  # ============================================================================

  @property
  def initialized(self) -> bool:
    """True if finalize() has been called, preventing new parameter creation."""
    return self._initialized

  # ============================================================================
  # Core API
  # ============================================================================

  def finalized(self) -> 'Params':
    """Returns finalized params that prevent new parameter creation.

    After finalization, attempting to create new parameters via get_param
    will raise KeyError. This catches bugs where parameter names change
    between training runs.
    """
    p = self._clone()
    p._initialized = True
    return p

  def split(
      self, predicate: Callable[[Path, Param], bool] | None = None
  ) -> tuple['Params', 'Params']:
    """Partitions parameters into two containers.

    Without arguments, splits into trainable and non-trainable parameters.
    This is the standard pattern for computing gradients:

      trainable, non_trainable = params.split()
      grads = jax.grad(loss_fn)(trainable, non_trainable, x)

    Args:
      predicate: Optional function (path, param) -> bool. Parameters where
        the predicate returns True go in the first container. Defaults to
        splitting by trainable flag.

    Returns:
      Tuple of (matching_params, non_matching_params).
    """

    def default_predicate(_: Path, p: Param) -> bool:
      return p.trainable

    if predicate is None:
      predicate = default_predicate

    match_data: dict[Path, Param] = {}
    other_data: dict[Path, Param] = {}
    for path, param in self._data.items():
      if predicate(path, param):
        match_data[path] = param
      else:
        other_data[path] = param

    match, other = self._clone(), self._clone()
    match._data, other._data = match_data, other_data
    return match, other

  def merge(self, other: 'Params') -> 'Params':
    """Combines this container with another.

    Parameters from `other` override those in `self` if paths conflict.
    Both containers must have the same initialized state.

    Args:
      other: Another Params container to merge in.

    Returns:
      A new merged Params container.

    Raises:
      ValueError: If initialized state doesn't match.
    """
    if self.initialized != other.initialized:
      raise ValueError(
          f'Initialized mismatch: {self.initialized} vs {other.initialized}.'
      )

    p = self._clone()
    p._data.update(other._data)
    return p

  # ============================================================================
  # Dict-like Access
  # ============================================================================

  def __getitem__(self, key: Path) -> Param:
    """Gets a parameter by its full path.

    Args:
      key: Tuple path like ('net', 'linear', 'kernel').

    Returns:
      The Param wrapper at that path.

    Raises:
      KeyError: If the path doesn't exist.
    """
    if key not in self._data:
      raise KeyError(f"Path '{key}' not found.")
    return self._data[key]

  def keys(self) -> KeysView[Path]:
    """Returns all parameter paths."""
    return self._data.keys()

  def values(self) -> ValuesView[Param]:
    """Returns all Param wrappers."""
    return self._data.values()

  def items(self) -> ItemsView[Path, Param]:
    """Returns (path, Param) pairs."""
    return self._data.items()

  def __len__(self) -> int:
    """Returns the number of parameters."""
    return len(self._data)

  def __contains__(self, key: Path) -> bool:
    """Returns True if the path exists in params."""
    return key in self._data

  # ============================================================================
  # Internal Methods
  # ============================================================================

  def _get(
      self,
      path: Path,
      shape: Shape,
      init: Initializer,
      dtype: jnp.dtype = jnp.float32,
      trainable: bool = True,
      metadata: dict[str, Any] | None = None,
      rng: 'Rng | None' = None,
  ) -> tuple[jax.Array, 'Params']:
    """Retrieves or creates a parameter. Use Module.get_param instead."""
    # Return existing parameter.
    if path in self._data:
      return self._data[path].value, self

    # Reject new params after finalization.
    if self._initialized:
      raise KeyError(f"Parameter '{path}' missing (params finalized).")

    # Generate a key if rng is provided. We always generate a key (incrementing
    # the counter) even if the initializer doesn't need it. This ensures
    # consistent initialization order: changing one param's initializer from
    # random to constant won't affect other params' random seeds.

    # NOTE: We bypass auto_fold_in_axes here intentionally. During param
    # initialization, we want identical params across all batch elements/devices.
    # auto_fold_in_axes is only for runtime operations like dropout.
    if rng is not None:
      key = rng.get_seed(self)
      counter = rng.get_counter(self)
      new_key = jax.random.fold_in(key, counter)
      new_p = rng.seed(self, counter=counter + 1)
      val = init(new_key, shape, dtype)
    else:
      new_p = self._clone()
      val = init(None, shape, dtype)  # type: ignore[arg-type]

    new_p._data[path] = Param(val, trainable=trainable, metadata=metadata)
    return val, new_p

  def _set(self, path: Path, value: Any) -> 'Params':
    """Updates a parameter value. Use Module.set_param instead."""
    if path not in self._data:
      raise KeyError(f"Path '{path}' not found.")

    p = self._clone()
    p._data[path] = self._data[path].replace(value=value)
    return p

  def _clone(self) -> 'Params':
    """Creates a shallow copy of this container."""
    p = cast(Params, object.__new__(Params))
    p._data = self._data.copy()
    p._initialized = self._initialized
    return p

  # ============================================================================
  # Special Methods
  # ============================================================================

  def __repr__(self) -> str:
    abstract_data = jax.eval_shape(lambda x: x, self._data)
    status = 'initialized' if self._initialized else 'uninitialized'
    lines = [f'Params[{status}]({{']
    for k, v in abstract_data.items():
      lines.append(f'  {k}: {v},')
    lines.append('})')
    return '\n'.join(lines)

  # ============================================================================
  # JAX Pytree Registration
  # ============================================================================

  def tree_flatten(self) -> tuple[tuple[dict[Path, Param]], tuple[bool]]:
    """Flattens for JAX pytree operations."""
    return (self._data,), (self._initialized,)

  @classmethod
  def tree_unflatten(
      cls,
      aux: tuple[bool],
      children: tuple[dict[Path, Param]],
  ) -> 'Params':
    """Unflattens from JAX pytree operations."""
    p = cast(Params, object.__new__(cls))
    p._data = children[0]
    p._initialized = aux[0]
    return p


jax.tree_util.register_pytree_node(
    Params, Params.tree_flatten, Params.tree_unflatten
)


# ==============================================================================
# Base Modules
# ==============================================================================


class Module:
  """Base class for neural network layers.

  Module provides the foundation for building neural network layers in blox.
  It connects layers to the Graph hierarchy for parameter namespacing and
  provides helper methods for parameter creation.

  Key features:
  - **Graph binding**: Each module owns a Graph node that namespaces its params.
  - **Constructor capture**: Arguments are automatically saved to graph metadata
    for visualization and serialization.
  - **Parameter helpers**: `get_param` and `set_param` simplify parameter
    handling.

  All subclasses must:
  1. Accept `graph` as the first constructor argument
  2. Call `super().__init__(graph)` in their `__init__`
  3. Implement `__call__(self, params, ...) -> (output, params)`

  Example:
    class Linear(bx.Module):
      def __init__(self, graph, output_size):
        super().__init__(graph)
        self.output_size = output_size

      def __call__(self, params, x):
        kernel, params = self.get_param(
            params, 'kernel', (x.shape[-1], self.output_size),
            jax.nn.initializers.lecun_normal()
        )
        return x @ kernel, params

    graph = bx.Graph('net')
    linear = Linear(graph.child('linear'), output_size=32)
  """

  # Temporary attributes used during __init__ to capture constructor arguments.
  # Set by __init_subclass__ wrapper, deleted after flushing to graph metadata.
  _blox_captured_args: dict[str, Any]
  _blox_captured_type: str

  def __init__(self, graph: Graph) -> None:
    """Binds this module to a graph node.

    Args:
      graph: A Graph node for namespacing this module's parameters.
        Must not be a root node - use `graph.child('name')` to create one.

    Raises:
      ValueError: If graph is a root node or already owned by another module.
    """
    if graph._is_root:
      raise ValueError(
          f"Cannot bind '{self.__class__.__name__}' to root graph node "
          f"'{graph.name}'. Use graph.child('name') to create a child node."
      )
    if '__type__' in graph.metadata:
      raise ValueError(
          f"Graph node '{graph.name}' already owned by "
          f"'{graph.metadata['__type__']}'. "
          f"Did you forget to call graph.child('name')?"
      )
    self.graph = graph

  def __init_subclass__(cls, **kwargs: Any) -> None:
    """Wraps subclass __init__ to capture constructor arguments.

    This metaclass-like hook automatically captures constructor arguments
    and stores them in the graph's metadata. This enables:
    - Visualization (bx.display shows constructor args)
    - Serialization (can reconstruct modules from metadata)
    - Debugging (easy to inspect what was passed)
    """
    super().__init_subclass__(**kwargs)

    original_init = cls.__init__

    # Skip wrapping if no __init__ or can't get signature.
    try:
      sig = inspect.signature(original_init)
    except ValueError:
      return

    @functools.wraps(original_init)
    def wrapped_init(self: 'Module', *args: Any, **kwargs: Any) -> None:
      # Capture args only at the outermost class in inheritance chain.
      # This ensures child class args take precedence over parent class args.
      should_capture = not hasattr(self, '_blox_captured_args')
      if should_capture:
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        self._blox_captured_args = {
            k: v
            for k, v in bound.arguments.items()
            if k not in {'self', 'graph', '__class__'}
        }
        self._blox_captured_type = cls.__name__

      original_init(self, *args, **kwargs)

      # Verify super().__init__ was called.
      if getattr(self, 'graph', None) is None:
        raise RuntimeError(
            f"Module '{cls.__name__}' failed to initialize. "
            'Did you forget to call super().__init__(graph)?'
        )

      # Flush captured data to graph metadata.
      if should_capture:
        self.graph.metadata['__type__'] = self._blox_captured_type
        self.graph.metadata.update(self._blox_captured_args)
        del self._blox_captured_args
        del self._blox_captured_type

    cls.__init__ = wrapped_init

  # ============================================================================
  # Parameter Access
  # ============================================================================

  def get_param(
      self,
      params: Params,
      name: str,
      shape: Shape,
      init: Initializer,
      dtype: jnp.dtype = jnp.float32,
      trainable: bool = True,
      metadata: dict[str, Any] | None = None,
      rng: 'Rng | None' = None,
  ) -> tuple[jax.Array, Params]:
    """Gets or creates a parameter in this module's namespace.

    On first call, creates a new parameter using the initializer.
    On subsequent calls, returns the existing parameter value.

    Args:
      params: The parameter container.
      name: Local parameter name (e.g., 'kernel', 'bias').
      shape: Shape of the parameter tensor.
      init: JAX initializer function.
      dtype: Data type (default: float32).
      trainable: Whether gradients should be computed (default: True).
      metadata: Optional metadata dict. Common keys:
        - 'sharding': tuple of mesh axis names for model parallelism.
      rng: Optional Rng module for stochastic initialization.

    Returns:
      Tuple of (parameter_value, updated_params).

    Example:
      kernel, params = self.get_param(
          params, 'kernel', (in_size, out_size),
          jax.nn.initializers.lecun_normal(),
          rng=self.rng,
          metadata={'sharding': (None, 'model')}
      )
    """
    return params._get(
        self.param_path(name), shape, init, dtype, trainable, metadata, rng
    )

  def set_param(self, params: Params, name: str, value: Any) -> Params:
    """Updates a parameter value in this module's namespace.

    Args:
      params: The parameter container.
      name: Local parameter name.
      value: New value for the parameter.

    Returns:
      Updated Params container.
    """
    return params._set(self.param_path(name), value)

  def param_path(self, name: str) -> Path:
    """Returns the full path for a parameter in this module's namespace.

    Args:
      name: Local parameter name.

    Returns:
      Full tuple path like ('net', 'linear', 'kernel').

    Example:
      # Check if a param exists.
      if module.param_path('kernel') in params:
        kernel, params = module.get_param(...)
    """
    return self.graph.path + (name,)

  # ============================================================================
  # Special Methods
  # ============================================================================

  def __repr__(self) -> str:
    """Returns a string showing the module type and constructor args."""
    if hasattr(self, 'graph') and hasattr(self.graph, 'metadata'):
      name = self.graph.metadata.get('__type__', self.__class__.__name__)
      args = [
          f'{k}={v!r}'
          for k, v in self.graph.metadata.items()
          if k != '__type__'
      ]
      return f"{name}({', '.join(args)})"
    return f'{self.__class__.__name__}()'

  @abc.abstractmethod
  def __call__(
      self,
      params: Params,
      *args: Any,
      **kwargs: Any,
  ) -> tuple[Any, Params]:
    """Applies the module to inputs.

    All subclasses must implement this method. The signature varies by module
    type, but the first argument is always `params` and the return value is
    always `(output, updated_params)`.
    """


def _validate_no_unnamed_axes(trace: Any) -> None:
  """Validates that no unnamed vmaps exist in the trace stack.

  Walks up the trace parent chain and raises an error if any unnamed vmap
  is found. Stops at JIT barriers since outer axes are irrelevant to
  compiled code.

  Raises:
    ValueError: If an unnamed vmap is detected.
  """
  if not hasattr(trace, 'parent_trace'):
    return

  trace_type = type(trace).__name__

  # Fail if unnamed vmap found.
  if trace_type == 'BatchTrace':
    if not isinstance(trace.axis_data.name, str):
      raise ValueError(
          'Unnamed vmap detected. Please provide `axis_name` to all `jax.vmap` '
          'calls that generate random keys using Rng with auto_fold_in_axes.'
      )

  # Stop at JIT barrier (outer axes are irrelevant to compiled code).
  if trace_type == 'DynamicJaxprTrace':
    return

  _validate_no_unnamed_axes(trace.parent_trace)


def _auto_axis_index() -> jax.Array | None:
  """Computes a unique index for the current position in vmap/shard_map.

  Returns None if not inside any vmap/shard_map. Otherwise returns an array
  that uniquely identifies this position across all axes.

  Raises:
    ValueError: If an unnamed vmap is detected.
  """
  # Validate all vmaps have names.
  _validate_no_unnamed_axes(jax.core.trace_ctx.trace)

  # Get all active axis names (logical order: outer -> inner).
  axis_names = jax.core.unsafe_get_axis_names_DO_NOT_USE()
  if not axis_names:
    return None

  return jax.lax.axis_index(tuple(axis_names))


class Rng(Module):
  """A random number generator stream stored as non-trainable params.

  Produces deterministic, counter-based random keys. The seed is stored in
  Params, not the Rng, which allows the same Rng module to be used with
  different seeds without changing the model structure.

  Seeds defined as int are converted and stored as a JAX key array.

  Example:
    graph = bx.Graph('net')
    rng = bx.Rng(graph.child('rng'))
    model = MyModel(graph.child('model'), rng=rng)

    # Create params and seed the Rng.
    params = rng.seed(bx.Params(), seed=42)

    # Forward pass creates parameters.
    _, params = model(params, x)
    params = params.finalized()

  Modules that need randomness should accept an Rng on construction:

    class Dropout(bx.Module):
      def __init__(self, graph, rate, rng):
        super().__init__(graph)
        self.rate = rate
        self.rng = rng

      def __call__(self, params, x, is_training=True):
        if not is_training:
          return x, params
        key, params = self.rng(params)
        return jax.random.dropout(key, self.rate, x), params
  """

  def __init__(self, graph: Graph, auto_fold_in_axes: bool = True) -> None:
    """Initializes the Rng module.

    Args:
      graph: The graph node for this module's scope.
      auto_fold_in_axes: If True (default), automatically folds in axis indices
        when inside shard_map/vmap to produce device-unique keys. This is not
        applied during parameter initialization so that parameters are not
        different on different devices. For sharded initialization, use jax.jit
        to handle Rng partitioning automatically, or handle folding manually
        with shard_map.
    """
    super().__init__(graph)
    self.auto_fold_in_axes = auto_fold_in_axes

  def get_seed(self, params: Params) -> jax.Array:
    """Returns the seed key.

    The seed is stored internally as a JAX key array.

    Args:
      params: The params container.

    Raises:
      KeyError: If this Rng is not initialized.
    """
    path = self.param_path('seed')
    if path not in params:
      raise KeyError(
          f"Rng '{self.graph.path}' not initialized. "
          'Use rng.seed(params, seed=...) first.'
      )
    return params[path].value

  def get_counter(self, params: Params) -> jax.Array:
    """Returns the counter value.

    Args:
      params: The params container.

    Raises:
      KeyError: If this Rng is not initialized.
    """
    path = self.param_path('counter')
    if path not in params:
      raise KeyError(
          f"Rng '{self.graph.path}' not initialized. "
          'Use rng.seed(params, seed=...) first.'
      )
    return params[path].value

  def seed(
      self,
      params: Params,
      *,
      seed: int | jax.Array | None = None,
      counter: int | jax.Array | None = None,
  ) -> Params:
    """Sets the seed and/or counter for this Rng.

    If this Rng is not yet initialized, creates the params with the given
    seed (required) and counter (defaults to 0). If already initialized,
    updates the specified values.

    Args:
      params: The params container.
      seed: Seed value (int or JAX key). Required if not initialized.
      counter: Counter value. Defaults to 0 if initializing, unchanged when None
        while updating.

    Returns:
      Updated params.

    Raises:
      ValueError: If not initialized and seed is None, or if initialized
        and both seed and counter are None.
    """
    is_init = self.param_path('seed') not in params

    if is_init:
      # Initializing: seed is required, counter defaults to 0.
      if seed is None:
        raise ValueError(
            f"Rng '{self.graph.path}' not initialized. Seed is required."
        )
      if counter is None:
        counter = 0
    else:
      # Updating: at least one must be provided.
      if seed is None and counter is None:
        raise ValueError('At least one of seed or counter must be provided.')

    if seed is not None:
      key = jax.random.key(seed) if isinstance(seed, int) else seed
      if is_init:
        key_init = jax.nn.initializers.constant(key, key.dtype)
        _, params = self.get_param(
            params,
            'seed',
            key.shape,
            key_init,
            key.dtype,
            trainable=False,
            metadata={'tag': 'rng_seed'},
        )
      else:
        params = self.set_param(params, 'seed', key)

    if counter is not None:
      counter_val = jnp.uint32(counter)
      if is_init:
        counter_init = jax.nn.initializers.constant(
            counter_val, counter_val.dtype
        )
        _, params = self.get_param(
            params,
            'counter',
            counter_val.shape,
            counter_init,
            counter_val.dtype,
            trainable=False,
            metadata={'tag': 'rng_counter'},
        )
      else:
        params = self.set_param(params, 'counter', counter_val)

    return params

  def __call__(self, params: Params) -> tuple[jax.Array, Params]:
    """Returns (new_key, new_params) tuple, with params' counter incremented.

    When auto_fold_in_axes is True (default), automatically folds in axis
    indices when inside shard_map and vmap to produce device-unique keys.

    For manual control over folding, set auto_fold_in_axes=False and use
    get_seed/seed directly:
      original_seed = rng.get_seed(params)
      folded_seed = jax.random.fold_in(
          original_seed, jax.lax.axis_index('batch')
      )
      params = rng.seed(params, seed=folded_seed)
      # ... do operations ...
      params = rng.seed(params, seed=original_seed)  # Restore before returning.

    Args:
      params: The params container.

    Raises:
      KeyError: If this Rng is not initialized. Use rng.seed(params, seed=...)
        first.
    """
    # Get seed, optionally with auto-folding.
    key = self.get_seed(params)
    if self.auto_fold_in_axes:
      axis_index = _auto_axis_index()
      if axis_index is not None:
        key = jax.random.fold_in(key, axis_index)

    counter = self.get_counter(params)

    # Fold in the counter to obtain a new deterministic key and increment.
    new_key = jax.random.fold_in(key, counter)
    params = self.seed(params, counter=counter + 1)

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
    step_fn: A callable that processes a single time step.
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
    step_fn: A callable that processes a single time step.
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
    return static_scan(
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

  @abc.abstractmethod
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

  @abc.abstractmethod
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

  @abc.abstractmethod
  def apply(
      self,
      params: Params,
      inputs: InputsT,
      prev_state: StateT | None = None,
      is_reset: ResetT | None = None,
      is_training: bool = True,
  ) -> tuple[tuple[OutputsT, StateT], Params]:
    """Processes a sequence of data [Batch, Time, ...].

    This method processes entire sequences, either step-by-step, which is the
    default behavior of RNN modules (see RecurrenceBase below), or in full,
    which is the default behavior of Transformer and Attention modules.

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


class RecurrenceBase(SequenceBase[InputsT, StateT, OutputsT, ResetT]):
  """Base class for Recurrent Neural Networks (RNNs).

  Implements sequence processing `apply` by applying the `__call__` method
  step-by-step (using either static or dynamic scan).

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
          lambda i, p, r: jnp.where(r, i, p),
          initial_state,
          prev_state,
          is_reset,
      )
    return cast(StateT, state)

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
