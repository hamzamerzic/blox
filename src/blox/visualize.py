"""Visualization utilities for blox models.

Integrates with Treescope for interactive inspection.
"""

from __future__ import annotations
from typing import Any

import treescope

from .interfaces import Graph, Module, Param, Params


class Leaf:
  """Visual wrapper for params."""

  def __init__(self, param: Param) -> None:
    self.param = param

  def __treescope_repr__(self, path: str, subtree_renderer: Any) -> Any:
    attr = {}

    if hasattr(self.param.value, 'shape'):
      attr['shape'] = self.param.value.shape
      attr['dtype'] = str(self.param.value.dtype)

    if self.param.metadata:
      attr['metadata'] = self.param.metadata

    attr['value'] = self.param.value

    status = '[T]' if self.param.trainable else '[N]'

    return treescope.repr_lib.render_object_constructor(
      object_type=type(f'Param{status}', (), {}),
      attributes=attr,
      path=path,
      subtree_renderer=subtree_renderer,
      roundtrippable=False,
    )


class Link:
  """Visual wrapper for dependency paths."""

  def __init__(self, path: tuple[str, ...]) -> None:
    self.path = path

  def __treescope_repr__(self, path: str, subtree_renderer: Any) -> Any:
    # Format tuple path as string for display.
    path_str = '/'.join(self.path)
    return treescope.repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={'path': path_str},
      path=path,
      subtree_renderer=subtree_renderer,
      roundtrippable=False,
    )


class NodeView:
  """Intermediate representation for visualization."""

  def __init__(
    self,
    typename: str,
    config: dict[str, Any],
    params: dict[str, Param],
    modules: dict[str, NodeView],
  ) -> None:
    self.typename = typename
    self.config = config
    self.params = params
    self.modules = modules

    self.num_params = 0
    self.bytes = 0
    self.total_params: int = 0

    for p in params.values():
      if hasattr(p.value, 'size'):
        self.num_params += p.value.size
      if hasattr(p.value, 'nbytes'):
        self.bytes += p.value.nbytes

    self.bytes += sum(m.bytes for m in modules.values())
    self.total_params = self.num_params + sum(
      m.total_params for m in modules.values()
    )

  def _format_size(self, size_bytes: int) -> str:
    if size_bytes < 1024:
      return f'{size_bytes} B'
    return f'{size_bytes / 1024:.1f} KB'

  def _clean_config_val(self, val: Any) -> Any:
    if isinstance(val, Module):
      return Link(path=val.graph.path)
    return val

  def __treescope_repr__(self, path: str, subtree_renderer: Any) -> Any:
    flat_dict: dict[str, Any] = {}

    for k, v in self.config.items():
      flat_dict[k] = self._clean_config_val(v)
    for k, v in self.params.items():
      flat_dict[k] = Leaf(v)
    for k, v in self.modules.items():
      flat_dict[k] = v

    title = f'{self.typename}'
    if self.total_params > 0:
      stats = f' # Param: {self.total_params} ({self._format_size(self.bytes)})'
      title += stats

    return treescope.repr_lib.render_object_constructor(
      object_type=type(title, (), {}),
      attributes=flat_dict,
      path=path,
      subtree_renderer=subtree_renderer,
      roundtrippable=False,
    )


def _view(graph: Graph, params: Params, is_root: bool = True) -> NodeView:
  """Internal helper to recursively build the NodeView."""
  my_params = {}
  # Access private data for visualization purposes.
  # pylint: disable=protected-access
  for key, value in params._data.items():
    # Check if this param belongs to this graph node (direct child).
    # key is a tuple like ('root', 'linear', 'w')
    # graph.path is a tuple like ('root', 'linear')
    # We want params where key[:-1] == graph.path (the param's parent is this node)
    if len(key) > 0 and key[:-1] == graph.path:
      param_name = key[-1]
      my_params[param_name] = value

  my_modules = {}
  for name, child_node in graph._children.items():
    my_modules[name] = _view(child_node, params, is_root=False)

  # Use the newly added metadata field safely.
  typename = graph.metadata.get('__type__', 'Graph')
  if is_root:
    typename = f'{graph.name}: {typename}'

  clean_config = {k: v for k, v in graph.metadata.items() if k != '__type__'}
  return NodeView(typename, clean_config, my_params, my_modules)


def display(graph: Graph, params: Params) -> None:
  """Builds the view and renders it with Treescope."""
  view = _view(graph, params, is_root=True)
  treescope.show(view)
