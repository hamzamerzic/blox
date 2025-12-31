"""Tests for path handling with special characters in names."""

import blox as bx
import jax
import jax.numpy as jnp


def test_slash_in_module_name():
  """Verifies that module names can contain '/' characters."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  # Module name with slash - was previously problematic.
  layer = bx.Linear(graph.child('encoder/decoder'), output_size=10, rng=rng)

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)

  y, params = layer(params, x)
  params = params.locked()

  assert y.shape == (2, 10)
  # Path should be a tuple with the slash preserved in the name.
  assert ('root', 'encoder/decoder', 'kernel') in params._data
  assert ('root', 'encoder/decoder', 'bias') in params._data


def test_slash_in_variable_name():
  """Verifies that variable names can contain '/' characters."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))

  class CustomModule(bx.Module):

    def __init__(self, g):
      super().__init__(g)

    def __call__(self, params, x):
      # Variable name with slash.
      w, params = self.get_param(
          params, 'weight/bias', (x.shape[-1], 10), jax.nn.initializers.zeros
      )
      return x @ w, params

  layer = CustomModule(graph.child('custom'))

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)

  _, params = layer(params, x)
  params = params.locked()

  # The slash should be preserved in the variable name.
  assert ('root', 'custom', 'weight/bias') in params._data


def test_special_characters_in_names():
  """Verifies various special characters work in module/variable names."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))

  special_names = [
      'layer.1',
      'block[0]',
      'attention:heads',
      'norm-pre',
      'fc_1',
      'émbed',  # Unicode
      'layer 1',  # Space
  ]

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)

  for name in special_names:
    layer = bx.Linear(graph.child(name), output_size=3, rng=rng)
    _, params = layer(params, x)

  params = params.locked()

  # All should be present.
  for name in special_names:
    assert ('root', name, 'kernel') in params._data, f'Failed for name: {name}'


def test_nested_slashes():
  """Verifies deeply nested paths with slashes in names."""
  graph = bx.Graph('model/v1')
  rng = bx.Rng(graph.child('rng'))
  child1 = graph.child('encoder/layer')
  child2 = child1.child('attention/head')
  layer = bx.Linear(child2.child('proj/out'), output_size=5, rng=rng)

  x = jnp.ones((2, 3))
  params = rng.seed(bx.Params(), seed=0)

  _, params = layer(params, x)
  params = params.locked()

  # Full path with all slashes preserved.
  expected_path = (
      'model/v1',
      'encoder/layer',
      'attention/head',
      'proj/out',
      'kernel',
  )
  assert expected_path in params._data


def test_split_with_special_characters():
  """Verifies split() works correctly with special character names."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  layer = bx.Linear(graph.child('layer/1'), output_size=3, rng=rng)

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)

  _, params = layer(params, x)
  params = params.locked()

  # Split should work - the predicate receives proper tuple paths.
  trainable, non_trainable = params.split()

  # Trainable should have the layer params.
  assert ('root', 'layer/1', 'kernel') in trainable._data
  assert ('root', 'layer/1', 'bias') in trainable._data

  # Non-trainable should have RNG (stored under Rng module's graph path).
  assert ('root', 'rng', 'seed') in non_trainable._data
  assert ('root', 'rng', 'counter') in non_trainable._data


def test_graph_repr_with_special_characters():
  """Verifies Graph repr handles special characters correctly."""
  graph = bx.Graph('model/v1')
  child = graph.child('layer/1')

  # Repr should format as path string with /.
  repr_str = repr(child)
  assert 'model/v1/layer/1' in repr_str


def test_custom_split():
  """Verifies Graph repr handles special characters correctly."""
  graph = bx.Graph('model/v1')
  rng = bx.Rng(graph.child('rng'))
  layer = bx.Linear(graph.child('layer/1'), output_size=3, rng=rng)

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)
  _, params = layer(params, x)
  params = params.locked()

  kernel, rest = params.split(lambda path, param: path[-1] == 'kernel')

  assert len(kernel._data) == 1
  assert ('model/v1', 'layer/1', 'kernel') in kernel._data

  assert ('model/v1', 'layer/1', 'bias') in rest._data
  assert ('model/v1', 'rng', 'seed') in rest._data
  assert ('model/v1', 'rng', 'counter') in rest._data


# =============================================================================
# Graph Traversal Tests
# =============================================================================


def test_graph_parent():
  """Verifies parent property returns the parent graph node."""
  graph = bx.Graph('root')
  child = graph.child('child')
  grandchild = child.child('grandchild')

  assert graph.parent is None
  assert child.parent is graph
  assert grandchild.parent is child


def test_graph_root():
  """Verifies root property returns the root of the hierarchy."""
  graph = bx.Graph('root')
  child = graph.child('child')
  grandchild = child.child('grandchild')

  assert graph.root is graph
  assert child.root is graph
  assert grandchild.root is graph


def test_graph_module_binding():
  """Verifies module property returns the bound module."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  child = graph.child('linear')
  layer = bx.Linear(child, output_size=10, rng=rng)

  # Root has no module.
  assert graph.module is None
  # Rng node has the Rng module.
  assert graph._children['rng'].module is rng
  # Linear node has the Linear module.
  assert child.module is layer


def test_graph_walk_simple():
  """Verifies walk() yields all descendant modules."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  _linear1 = bx.Linear(graph.child('linear1'), output_size=10, rng=rng)
  _linear2 = bx.Linear(graph.child('linear2'), output_size=5, rng=rng)

  modules = list(graph.walk())

  assert len(modules) == 3  # rng, linear1, linear2
  paths = [path for path, _ in modules]
  assert ('root', 'rng') in paths
  assert ('root', 'linear1') in paths
  assert ('root', 'linear2') in paths


def test_graph_walk_nested():
  """Verifies walk() traverses nested module hierarchies."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))

  # Build a nested structure: root -> encoder -> layer1, layer2
  encoder_graph = graph.child('encoder')

  class Encoder(bx.Module):

    def __init__(self, g, rng):
      super().__init__(g)
      self.layer1 = bx.Linear(g.child('layer1'), output_size=10, rng=rng)
      self.layer2 = bx.Linear(g.child('layer2'), output_size=5, rng=rng)

    def __call__(self, params, x):
      x, params = self.layer1(params, x)
      return self.layer2(params, x)

  _encoder = Encoder(encoder_graph, rng)

  modules = list(graph.walk())

  # Should find: rng, encoder, encoder/layer1, encoder/layer2
  assert len(modules) == 4
  paths = [path for path, _ in modules]
  assert ('root', 'rng') in paths
  assert ('root', 'encoder') in paths
  assert ('root', 'encoder', 'layer1') in paths
  assert ('root', 'encoder', 'layer2') in paths


def test_graph_walk_filter_by_type():
  """Verifies walk() can be filtered by module type."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  linear = bx.Linear(graph.child('linear'), output_size=10, rng=rng)
  bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)  # Not used directly.

  # Filter for Linear modules only.
  linears = [
      (path, mod) for path, mod in graph.walk() if isinstance(mod, bx.Linear)
  ]

  assert len(linears) == 1
  assert linears[0][1] is linear


def test_graph_double_bind_fails():
  """Verifies that binding a module to an already-bound node fails."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  child = graph.child('linear')
  _ = bx.Linear(child, output_size=10, rng=rng)

  # Trying to bind another module to the same node should fail.
  try:
    _ = bx.Linear(child, output_size=5, rng=rng)
    assert False, 'Should have raised ValueError'
  except ValueError as e:
    assert 'already bound' in str(e)
