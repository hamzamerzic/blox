"""Visualization utilities for blox models.

Integrates with Treescope for interactive inspection.
"""

from __future__ import annotations
from typing import Any
from treescope import repr_lib

from .interfaces import Graph, Module, Params, Variable


class Leaf:
  """Visual wrapper for variables."""

  def __init__(self, var: Variable) -> None:
    self.var = var

  def __treescope_repr__(self, path: str, subtree_renderer: Any) -> Any:
    attr = {}

    if hasattr(self.var.value, 'shape'):
      attr['shape'] = self.var.value.shape
      attr['dtype'] = str(self.var.value.dtype)

    if self.var.metadata:
      attr['metadata'] = self.var.metadata

    attr['value'] = self.var.value

    status = '[T]' if self.var.trainable else '[N]'

    return repr_lib.render_object_constructor(
      object_type=type(f'Param{status}', (), {}),
      attributes=attr,
      path=path,
      subtree_renderer=subtree_renderer,
      roundtrippable=False,
    )


class Link:
  """Visual wrapper for dependency paths."""

  def __init__(self, path: str) -> None:
    self.path = path

  def __treescope_repr__(self, path: str, subtree_renderer: Any) -> Any:
    return repr_lib.render_object_constructor(
      object_type=type(self),
      attributes={'path': self.path},
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
    params: dict[str, Variable],
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

    return repr_lib.render_object_constructor(
      object_type=type(title, (), {}),
      attributes=flat_dict,
      path=path,
      subtree_renderer=subtree_renderer,
      roundtrippable=False,
    )


def display(graph: Graph, params: Params, is_root: bool = True) -> NodeView:
  prefix = f'{graph.path}/' if graph.path else ''

  my_params = {}
  # Access private data for visualization purposes
  # pylint: disable=protected-access
  for key, value in params._data.items():
    if key.startswith(prefix) or (not prefix and key):
      local_name = key[len(prefix) :]
      if '/' not in local_name:
        my_params[local_name] = value

  # Special case: If we are at the root, show the rng variable.
  if is_root and 'rng' in params._data:
    my_params['rng'] = params._data['rng']

  my_modules = {}
  for name, child_node in graph._children.items():
    my_modules[name] = display(child_node, params, is_root=False)

  # Use the newly added metadata field safely
  typename = graph.metadata.get('__type__', 'Graph')
  if is_root:
    typename = f'{graph.name}: {typename}'

  clean_config = {k: v for k, v in graph.metadata.items() if k != '__type__'}
  return NodeView(typename, clean_config, my_params, my_modules)
