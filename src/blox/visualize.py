"""Interactive visualization for blox models using Treescope.

This module renders model structure and parameters as an interactive tree.
The main entry point is `display(graph, params)`.

Example:
  graph = bx.Graph('net')
  linear = bx.Linear(graph.child('linear'), output_size=32)
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))
  _, params = linear(params, x)
  bx.display(graph, params)
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


class ParamView:
  """Treescope wrapper for displaying a single parameter.

  Shows shape, dtype, trainable status, and value with statistics.
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


class ModuleRefView:
  """Treescope wrapper for a reference to another module.

  Used when a module has injected dependencies (e.g., MyModule(linear=linear))
  to show a path reference instead of duplicating the entire subtree.
  """

  def __init__(self, path: tuple[str, ...], typename: str) -> None:
    self.path = path
    self.typename = typename

  def __treescope_repr__(self, path: str, subtree_renderer: Any) -> Any:
    path_str = '/'.join(self.path)
    return treescope.repr_lib.render_object_constructor(
        object_type=type(f'→ {self.typename}', (), {}),
        attributes={'path': path_str},
        path=path,
        subtree_renderer=subtree_renderer,
        roundtrippable=False,
    )


class NodeView:
  """Treescope wrapper representing a module node in the visualization tree.

  Each NodeView shows:
  - Module type and total parameter count in the title
  - Constructor arguments (non-default, non-module values only)
  - Parameters at this node
  - Child modules
  - References to injected dependencies
  """

  def __init__(
      self,
      typename: str,
      config: dict[str, Any],
      params: dict[str, bx.Param],
      children: dict[str, 'NodeView'],
      refs: dict[str, ModuleRefView],
  ) -> None:
    self.typename = typename
    self.config = config
    self.params = params
    self.children = children
    self.refs = refs

    # Compute parameter statistics.
    self.num_params = 0
    self.num_bytes = 0
    for p in params.values():
      if hasattr(p.value, 'size'):
        self.num_params += p.value.size
      if hasattr(p.value, 'nbytes'):
        self.num_bytes += p.value.nbytes

    # Include children in totals.
    self.num_bytes += sum(c.num_bytes for c in children.values())
    self.total_params = self.num_params + sum(
        c.total_params for c in children.values()
    )

  def __treescope_repr__(self, path: str, subtree_renderer: Any) -> Any:
    # Title includes type and param count.
    title = self.typename
    if self.total_params > 0:
      title += (
          f' # {self.total_params:,} params ({_format_bytes(self.num_bytes)})'
      )

    body: dict[str, Any] = {}

    # Show non-trivial config values (skip None/defaults and module refs).
    if self.config:
      body['config'] = self.config

    # Show references to injected modules.
    for name, ref in self.refs.items():
      body[name] = ref

    # Show parameters at this node.
    for name, param in self.params.items():
      body[name] = ParamView(param)

    # Show child modules.
    for name, child in self.children.items():
      body[name] = child

    return treescope.repr_lib.render_object_constructor(
        object_type=type(title, (), {}),
        attributes=body,
        path=path,
        subtree_renderer=subtree_renderer,
        roundtrippable=False,
    )


def _build_tree(
    graph: bx.Graph,
    params: bx.Params,
    registry: dict[tuple[str, ...], NodeView],
) -> NodeView:
  """Recursively build the visualization tree from Graph and Params.

  Args:
    graph: Current graph node to visualize.
    params: Parameter container with all model state.
    registry: Maps graph paths to their NodeViews (for reference resolution).

  Returns:
    NodeView for this graph node and all its descendants.
  """
  # Collect parameters directly under this graph path.
  my_params: dict[str, bx.Param] = {}
  for key, param in params._data.items():
    if len(key) > 0 and key[:-1] == graph.path:
      my_params[key[-1]] = param

  # Recursively build children.
  my_children: dict[str, NodeView] = {}
  for name, child_graph in graph._children.items():
    my_children[name] = _build_tree(child_graph, params, registry)

  # Extract type and config from metadata.
  typename = graph.metadata.get('__type__', 'Graph')

  # Filter config: skip __type__, None values, and module references.
  config: dict[str, Any] = {}
  for k, v in graph.metadata.items():
    if k == '__type__':
      continue
    if v is None:
      continue
    if hasattr(v, 'graph'):
      continue  # Module references handled separately.
    config[k] = v

  # Build references for injected module dependencies.
  refs: dict[str, ModuleRefView] = {}

  view = NodeView(typename, config, my_params, my_children, refs)
  registry[graph.path] = view
  return view


def _resolve_refs(
    graph: bx.Graph,
    view: NodeView,
    registry: dict[tuple[str, ...], NodeView],
) -> None:
  """Add ModuleRefViews for injected module dependencies.

  When a module stores another module as an attribute (dependency injection),
  this creates a reference link instead of duplicating the subtree.

  Args:
    graph: Current graph node.
    view: Corresponding NodeView to add references to.
    registry: Maps graph paths to NodeViews for looking up references.
  """
  for key, value in graph.metadata.items():
    if key == '__type__':
      continue
    # Check if this is a module reference.
    if hasattr(value, 'graph') and hasattr(value.graph, 'path'):
      ref_path = value.graph.path
      if ref_path in registry:
        ref_typename = registry[ref_path].typename
        view.refs[key] = ModuleRefView(ref_path, ref_typename)

  # Recurse into children.
  for name, child_graph in graph._children.items():
    if name in view.children:
      _resolve_refs(child_graph, view.children[name], registry)


def display(graph: bx.Graph, params: bx.Params) -> None:
  """Display model structure and parameters as an interactive tree.

  Builds a visual tree showing:
  - Module hierarchy with type names
  - Parameter counts and memory usage
  - Constructor arguments (non-default values)
  - Parameter shapes, dtypes, and value statistics
  - References to injected module dependencies

  Args:
    graph: Root Graph node of the model.
    params: Params container with model state.

  Example:
    graph = bx.Graph('net')
    encoder = bx.Linear(graph / 'encoder', output_size=256)
    decoder = bx.Linear(graph / 'decoder', output_size=128)
    params = bx.Params(rng=bx.Rng(graph / 'rng', seed=42))
    _, params = encoder(params, x)
    _, params = decoder(params, encoder_out)
    bx.display(graph, params)
  """
  registry: dict[tuple[str, ...], NodeView] = {}
  view = _build_tree(graph, params, registry)

  # Prefix root with graph name.
  view.typename = f'{graph.name}: {view.typename}'

  # Resolve injected module references.
  _resolve_refs(graph, view, registry)

  treescope.show(view)
