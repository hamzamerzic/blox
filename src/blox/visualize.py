"""Interactive visualization for blox models using Treescope.

This module renders model structure and parameters as an interactive tree.
The main entry point is `display(graph, params)`.

Example:
  graph = bx.Graph('net')
  linear = bx.Linear(graph.child('linear'), output_size=32)
  params = rng.seed(bx.Params(), seed=42)
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


class ConstructorView:
  """Renders as ClassName(arg1=..., arg2=...).

  Treescope will automatically handle references if values in args are the
  same objects displayed elsewhere in the tree.
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


class NodeView:
  """Treescope wrapper representing a module node in the visualization tree.

  Each NodeView shows:
  - Module type and total parameter count in the title
  - Constructor arguments via __init__ (with references to other modules)
  - Parameters at this node
  - Child modules
  """

  def __init__(
      self,
      typename: str,
      config: dict[str, Any],
      params: dict[str, bx.Param],
      modules: dict[str, 'NodeView'],
  ) -> None:
    self.typename = typename
    self.config = config
    self.params = params
    self.modules = modules

    # Store ConstructorView for reference linking.
    # When another module references this one, we link to this object so
    # treescope renders it as a reference (same object in multiple places).
    self.constructor = ConstructorView(typename, config) if config else None

    # Compute parameter statistics.
    self.num_params = 0
    self.bytes = 0
    for p in params.values():
      if hasattr(p.value, 'size'):
        self.num_params += p.value.size
      if hasattr(p.value, 'nbytes'):
        self.bytes += p.value.nbytes

    self.bytes += sum(m.bytes for m in modules.values())
    self.total_params = self.num_params + sum(
        m.total_params for m in modules.values()
    )

  def __treescope_repr__(self, path: str, subtree_renderer: Any) -> Any:
    title = self.typename
    if self.total_params > 0:
      title += f' # Param: {self.total_params} ({_format_bytes(self.bytes)})'

    body: dict[str, Any] = {}

    # Constructor (__init__): shows the full config including module references.
    # After _link_dependencies, injected modules point to ConstructorView objects.
    # Treescope renders them as references since the same object exists here.
    if self.constructor:
      body['__init__'] = self.constructor

    for k, v in self.params.items():
      body[k] = ParamView(v)

    for k, v in self.modules.items():
      body[k] = v

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

  # Filter config: skip __type__ and None values.
  config: dict[str, Any] = {}
  for k, v in graph.metadata.items():
    if k == '__type__':
      continue
    if v is None:
      continue
    config[k] = v

  view = NodeView(typename, config, my_params, my_children)
  registry[graph.path] = view
  return view


def _link_dependencies(
    view: NodeView,
    registry: dict[tuple[str, ...], NodeView],
) -> None:
  """Replace module references in config with ConstructorView objects.

  When a module stores another module as an attribute (dependency injection),
  we replace the module object with the referenced module's ConstructorView.
  Treescope renders these as references since the same object appears in the
  referenced module's __init__ entry.

  Args:
    view: NodeView whose config may contain module references.
    registry: Maps graph paths to NodeViews for looking up references.
  """
  if view.constructor:
    for key, value in list(view.constructor.args.items()):
      if hasattr(value, 'graph') and hasattr(value.graph, 'path'):
        ref_path = value.graph.path
        if ref_path in registry and registry[ref_path].constructor:
          view.constructor.args[key] = registry[ref_path].constructor

  for child in view.modules.values():
    _link_dependencies(child, registry)


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
    params = rng.seed(bx.Params(), seed=42)
    _, params = encoder(params, x)
    _, params = decoder(params, encoder_out)
    bx.display(graph, params)
  """
  registry: dict[tuple[str, ...], NodeView] = {}
  view = _build_tree(graph, params, registry)

  # Prefix root with graph name.
  view.typename = f'{graph.name}: {view.typename}'

  # Link injected module references.
  _link_dependencies(view, registry)

  treescope.show(view)
