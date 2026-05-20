"""Interactive visualization for blox models using Treescope.

This module renders model structure and parameters as an interactive
tree. The main entry point is :func:`display`.

Example::

    graph = bx.Graph('net')
    rng = bx.Rng(graph.child('rng'))
    linear = bx.Linear(graph.child('linear'), output_size=32, rng=rng)
    params = rng.seed(bx.Params(), seed=42)
    _, params = linear(params, x)
    bx.display(graph, params)

    # Or without params (structure only):
    bx.display(graph)
"""

from __future__ import annotations

from typing import Any

import treescope

from . import interfaces as bx


def _format_bytes(n: int) -> str:
  """Format byte count as human-readable string."""
  if n < 1024:
    return f'{n} B'
  return f'{n / 1024:.1f} KB'


def _get_module_info(module: bx.Module | None) -> tuple[str, dict[str, Any]]:
  """Extract typename and config from a module.

  Module stores its type name and constructor args in _init_type and _init_args
  (set by __init_subclass__). We filter out None values from config since they
  add noise to the visualization.

  Returns:
    Tuple of (typename, config dict with non-None values).
  """
  if module is not None:
    typename = getattr(module, '_init_type', type(module).__name__)
    init_args = getattr(module, '_init_args', {})
    config = {k: v for k, v in init_args.items() if v is not None}
  else:
    typename = 'Graph'
    config = {}
  return typename, config


def _get_params_at_path(
    params: bx.Params | None, path: tuple[str, ...]
) -> dict[str, bx.Param]:
  """Collect parameters directly under a given path.

  Params are stored with tuple paths like ('net', 'linear', 'kernel').
  This function finds all params where the parent path matches the given path.

  Args:
    params: The Params container, or None.
    path: The path to match (e.g., ('net', 'linear')).

  Returns:
    Dict mapping param names to Param objects at that path.
  """
  result: dict[str, bx.Param] = {}
  if params is not None:
    for key, param in params.items():
      if len(key) > 0 and key[:-1] == path:
        result[key[-1]] = param
  return result


# =============================================================================
# View Classes
# =============================================================================


class ParamView:
  """Treescope wrapper for displaying a single parameter.

  Shows shape, dtype, trainable status [T]/[N], metadata, and value.
  """

  def __init__(self, param: bx.Param) -> None:
    self.param = param

  def __treescope_repr__(self, path: str, subtree_renderer: Any) -> Any:
    attrs: dict[str, Any] = {}
    if hasattr(self.param.value, 'shape'):
      attrs['shape'] = self.param.value.shape
      attrs['dtype'] = str(self.param.value.dtype)
    if self.param.metadata:
      attrs['metadata'] = self.param.metadata
    attrs['value'] = self.param.value

    # [T] = trainable, [N] = non-trainable.
    tag = '[T]' if self.param.trainable else '[N]'
    return treescope.repr_lib.render_object_constructor(
        object_type=type(f'Param{tag}', (), {}),
        attributes=attrs,
        path=path,
        subtree_renderer=subtree_renderer,
        roundtrippable=False,
    )


class ConstructorView:
  """Renders as ClassName(arg1=..., arg2=...).

  Used to display module constructor arguments. Treescope automatically
  handles references when the same object appears in multiple places.
  """

  def __init__(self, class_name: str, args: dict[str, Any]) -> None:
    self.name = class_name
    self.args = args

  def __treescope_repr__(self, path: str, subtree_renderer: Any) -> Any:
    return treescope.repr_lib.render_object_constructor(
        object_type=type(self.name, (), {}),
        attributes=self.args,
        path=path,
        subtree_renderer=subtree_renderer,
        roundtrippable=False,
    )


class MultiGraphView:
  """Wrapper for displaying multiple graphs as a single tree.

  When visualizing multiple graphs (e.g., encoder and decoder with shared
  components), this creates a single "Model" root with each graph as a child.
  """

  def __init__(self, graphs: dict[str, 'NodeView']) -> None:
    self.graphs = graphs
    self.total_params = sum(g.total_params for g in graphs.values())
    self.bytes = sum(g.bytes for g in graphs.values())

  def __treescope_repr__(self, path: str, subtree_renderer: Any) -> Any:
    title = 'Model'
    if self.total_params > 0:
      title += f' # Params: {self.total_params} ({_format_bytes(self.bytes)})'
    return treescope.repr_lib.render_object_constructor(
        object_type=type(title, (), {}),
        attributes=self.graphs,
        path=path,
        subtree_renderer=subtree_renderer,
        roundtrippable=False,
    )


class NodeView:
  """Wrapper representing a module node in the visualization tree.

  Each NodeView shows:
  - Module type and total parameter count in the title
  - Constructor arguments (via __init__ if params present, or directly if not)
  - Parameters at this node with shape/dtype/value
  - Child modules

  The constructor attribute stores a ConstructorView for reference linking.
  When module A references module B in its config, we replace the module
  object with B's ConstructorView so treescope renders it as a reference.
  """

  def __init__(
      self,
      typename: str,
      config: dict[str, Any],
      params: dict[str, bx.Param],
      children: dict[str, 'NodeView'],
      has_params: bool = True,
  ) -> None:
    self.typename = typename
    self.config = config
    self.params = params
    self.children = children
    self.has_params = has_params

    # For reference linking between modules. When another module stores this
    # one as an attribute, we link to this ConstructorView so treescope
    # renders it as a reference (same object in multiple places).
    self.constructor = ConstructorView(typename, config) if config else None

    # Compute parameter statistics for display in title.
    self.num_params = sum(
        p.value.size for p in params.values() if hasattr(p.value, 'size')
    )
    self.bytes = sum(
        p.value.nbytes for p in params.values() if hasattr(p.value, 'nbytes')
    )
    self.bytes += sum(c.bytes for c in children.values())
    self.total_params = self.num_params + sum(
        c.total_params for c in children.values()
    )

  def __treescope_repr__(self, path: str, subtree_renderer: Any) -> Any:
    title = self.typename
    if self.total_params > 0:
      title += f' # Params: {self.total_params} ({_format_bytes(self.bytes)})'

    body: dict[str, Any] = {}

    if self.has_params:
      # With params: show constructor args under __init__, then params.
      if self.constructor:
        body['__init__'] = self.constructor
      for k, v in self.params.items():
        body[k] = ParamView(v)
    else:
      # Without params: show config args directly (structure-only mode).
      for k, v in self.config.items():
        body[k] = v

    # Add child modules.
    for k, v in self.children.items():
      body[k] = v

    return treescope.repr_lib.render_object_constructor(
        object_type=type(title, (), {}),
        attributes=body,
        path=path,
        subtree_renderer=subtree_renderer,
        roundtrippable=False,
    )


# =============================================================================
# Tree Building
# =============================================================================


def _build_tree(
    graph: bx.Graph,
    params: bx.Params | None,
    registry: dict[tuple[str, ...], NodeView],
) -> NodeView:
  """Recursively build the visualization tree from a Graph.

  Walks the graph hierarchy, collecting parameters and building NodeViews.
  Each NodeView is registered by its path for later reference linking.

  Args:
    graph: Current graph node to visualize.
    params: Parameter container, or None for structure-only mode.
    registry: Maps graph paths to NodeViews (populated during traversal).

  Returns:
    NodeView for this graph node and all its descendants.
  """
  my_params = _get_params_at_path(params, graph.path)
  typename, config = _get_module_info(graph.module)

  # Recursively build children.
  children: dict[str, NodeView] = {}
  for name, child_graph in graph._children.items():
    children[name] = _build_tree(child_graph, params, registry)

  view = NodeView(typename, config, my_params, children, params is not None)
  registry[graph.path] = view
  return view


def _build_external_view(
    module: bx.Module,
    params: bx.Params | None,
) -> NodeView:
  """Build a minimal NodeView for an external module.

  External modules are those referenced by a module but belonging to a
  different graph hierarchy. We create a simple view with just the module's
  own params (no children, since we don't have the full graph).

  Args:
    module: The external module to visualize.
    params: Parameter container, or None for structure-only mode.

  Returns:
    NodeView for just this module (no children).
  """
  typename, config = _get_module_info(module)
  my_params = _get_params_at_path(params, module.graph.path)
  return NodeView(typename, config, my_params, {}, params is not None)


def _collect_external_modules(
    view: NodeView,
    registry: dict[tuple[str, ...], NodeView],
    external_registry: dict[tuple[str, ...], NodeView],
    params: bx.Params | None,
) -> None:
  """Find and collect modules referenced from outside the graph hierarchy.

  When a module stores another module as an attribute (dependency injection),
  that module might be from a different graph. This function finds such
  external references and builds minimal views for them.

  Args:
    view: NodeView whose config may contain external module references.
    registry: Maps paths within our graph to NodeViews.
    external_registry: Collects external modules found (populated in-place).
    params: Parameter container, or None for structure-only mode.
  """
  if view.constructor:
    for value in view.constructor.args.values():
      # Check if this value is a module with a graph path.
      if hasattr(value, 'graph') and hasattr(value.graph, 'path'):
        ref_path = value.graph.path
        # Skip if already in registry (same graph) or external_registry.
        if ref_path not in registry and ref_path not in external_registry:
          external_registry[ref_path] = _build_external_view(value, params)

  # Recurse into children.
  for child in view.children.values():
    _collect_external_modules(child, registry, external_registry, params)


def _link_dependencies(
    view: NodeView,
    registry: dict[tuple[str, ...], NodeView],
    external_registry: dict[tuple[str, ...], NodeView],
) -> None:
  """Replace module references with ConstructorView objects for linking.

  When a module stores another module as an attribute, we replace the module
  object in the config with the referenced module's ConstructorView. Treescope
  then renders these as references (showing the same object in multiple places).

  Args:
    view: NodeView whose config may contain module references.
    registry: Maps graph paths to NodeViews.
    external_registry: Maps external module paths to their NodeViews.
  """
  if view.constructor:
    for key, value in list(view.constructor.args.items()):
      if hasattr(value, 'graph') and hasattr(value.graph, 'path'):
        ref_path = value.graph.path
        # Look up in registry first, then external_registry.
        for reg in (registry, external_registry):
          if ref_path in reg and reg[ref_path].constructor:
            view.constructor.args[key] = reg[ref_path].constructor
            break

  # Recurse into children.
  for child in view.children.values():
    _link_dependencies(child, registry, external_registry)


# =============================================================================
# Public API
# =============================================================================


def display(
    graph: bx.Graph | tuple[bx.Graph, ...],
    params: bx.Params | None = None,
) -> None:
  """Display model structure and parameters as an interactive tree.

  Builds a visual tree showing:
  - Module hierarchy with type names
  - Parameter counts and memory usage (if params provided)
  - Constructor arguments
  - Parameter shapes, dtypes, and value statistics (if params provided)
  - References between modules (dependency injection)

  Args:
    graph: Root Graph node(s). Pass a tuple to display multiple graphs
      together in a single view.
    params: Optional Params container. If None, shows only the module
      hierarchy and constructor arguments (structure-only mode).

  Example::

      # Full display with params:
      bx.display(graph, params)

      # Structure only (no params):
      bx.display(graph)

      # Multiple graphs in one view:
      bx.display((encoder_graph, decoder_graph), params)
  """
  # Normalize single graph to tuple.
  graphs = (graph,) if isinstance(graph, bx.Graph) else graph

  registry: dict[tuple[str, ...], NodeView] = {}
  external_registry: dict[tuple[str, ...], NodeView] = {}
  views: list[NodeView] = []
  names: list[str] = []

  # Build views for each graph.
  for g in graphs:
    views.append(_build_tree(g, params, registry))
    names.append(g.name)

  # Collect external module references (modules from other graphs).
  for view in views:
    _collect_external_modules(view, registry, external_registry, params)

  # Group external modules by their root graph name.
  # E.g., if net2.rng is referenced, we group it under 'net2'.
  external_roots: dict[str, dict[str, NodeView]] = {}
  for path, ext_view in external_registry.items():
    root = path[0]
    if root not in external_roots:
      external_roots[root] = {}
    # Use the last path element as the child name.
    external_roots[root][path[-1] if len(path) > 1 else root] = ext_view

  # Create container views for external graph roots.
  for root_name, modules in external_roots.items():
    ext_root = NodeView('(external)', {}, {}, modules, params is not None)
    # Manually set totals since NodeView computes from children.
    ext_root.total_params = sum(m.total_params for m in modules.values())
    ext_root.bytes = sum(m.bytes for m in modules.values())
    views.append(ext_root)
    names.append(root_name)
    registry[(root_name,)] = ext_root

  # Link module references (replace Module objects with ConstructorViews).
  for view in views:
    _link_dependencies(view, registry, external_registry)

  # Display the tree(s).
  if len(views) == 1:
    # Single graph: prefix typename with graph name for context.
    views[0].typename = f'{names[0]}: {views[0].typename}'
    treescope.show(views[0])
  else:
    # Multiple graphs: combine into single tree with "Model" root.
    treescope.show(MultiGraphView(dict(zip(names, views))))
