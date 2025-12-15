"""Tests for path handling with special characters in names."""

import jax
import jax.numpy as jnp
import blox as bx


def test_slash_in_module_name():
  """Verifies that module names can contain '/' characters."""
  graph = bx.Graph('root')
  # Module name with slash - was previously problematic.
  layer = bx.Linear(graph.child('encoder/decoder'), output_size=10)

  x = jnp.ones((2, 5))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, params = layer(params, x)
  params = params.finalize()

  assert y.shape == (2, 10)
  # Path should be a tuple with the slash preserved in the name.
  assert ('root', 'encoder/decoder', 'kernel') in params._data
  assert ('root', 'encoder/decoder', 'bias') in params._data


def test_slash_in_variable_name():
  """Verifies that variable names can contain '/' characters."""
  graph = bx.Graph('root')

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
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  _, params = layer(params, x)
  params = params.finalize()

  # The slash should be preserved in the variable name.
  assert ('root', 'custom', 'weight/bias') in params._data


def test_special_characters_in_names():
  """Verifies various special characters work in module/variable names."""
  graph = bx.Graph('root')

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
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  for name in special_names:
    layer = bx.Linear(graph.child(name), output_size=3)
    _, params = layer(params, x)

  params = params.finalize()

  # All should be present.
  for name in special_names:
    assert ('root', name, 'kernel') in params._data, f'Failed for name: {name}'


def test_nested_slashes():
  """Verifies deeply nested paths with slashes in names."""
  graph = bx.Graph('model/v1')
  child1 = graph.child('encoder/layer')
  child2 = child1.child('attention/head')
  layer = bx.Linear(child2.child('proj/out'), output_size=5)

  x = jnp.ones((2, 3))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  _, params = layer(params, x)
  params = params.finalize()

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
  layer = bx.Linear(graph.child('layer/1'), output_size=3)

  x = jnp.ones((2, 5))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  _, params = layer(params, x)
  params = params.finalize()

  # Split should work - the predicate receives proper tuple paths.
  trainable, non_trainable = params.split()

  # Trainable should have the layer params.
  assert ('root', 'layer/1', 'kernel') in trainable._data
  assert ('root', 'layer/1', 'bias') in trainable._data

  # Non-trainable should have RNG (stored under Rng module's graph path).
  assert ('root', 'rng', 'key') in non_trainable._data
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
  layer = bx.Linear(graph.child('layer/1'), output_size=3)

  x = jnp.ones((2, 5))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))
  _, params = layer(params, x)
  params = params.finalize()

  kernel, rest = params.split(lambda path, param: path[-1] == 'kernel')

  assert len(kernel._data) == 1
  assert ('model/v1', 'layer/1', 'kernel') in kernel._data

  assert ('model/v1', 'layer/1', 'bias') in rest._data
  assert ('model/v1', 'rng', 'key') in rest._data
  assert ('model/v1', 'rng', 'counter') in rest._data
