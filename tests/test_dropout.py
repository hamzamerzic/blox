import jax.numpy as jnp
import blox as bx
import pytest


def test_dropout_training_mode():
  """Verifies that dropout zeros some elements during training."""
  graph = bx.Graph('root')
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5)

  x = jnp.ones((100, 100))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))

  y, params = dropout(params, x, is_training=True)

  # Some elements should be zero.
  num_zeros = jnp.sum(y == 0.0)
  assert num_zeros > 0, 'Dropout should zero some elements.'

  # Not all elements should be zero.
  assert num_zeros < x.size, 'Dropout should not zero all elements.'


def test_dropout_inference_mode():
  """Verifies that dropout is identity during inference."""
  graph = bx.Graph('root')
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5)

  x = jnp.ones((10, 10))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))

  y, _ = dropout(params, x, is_training=False)

  # Output should be identical to input.
  assert jnp.allclose(y, x), 'Dropout should be identity during inference.'


def test_dropout_rng_consumption():
  """Verifies that dropout consumes RNG only during training."""
  graph = bx.Graph('root')
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5)

  x = jnp.ones((10, 10))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))

  # RNG counter is stored under the Rng module's graph path.
  counter_path = ('root', 'rng', 'counter')

  # Training mode should consume RNG (counter gets initialized on first call).
  _, params_train = dropout(params, x, is_training=True)
  counter_after_train = params_train._data[counter_path].value

  # Second training call should increment counter.
  _, params_train2 = dropout(params_train, x, is_training=True)
  assert params_train2._data[counter_path].value == counter_after_train + 1

  # Inference mode should not consume RNG.
  _, params_infer = dropout(params_train, x, is_training=False)
  assert params_infer._data[counter_path].value == counter_after_train


def test_dropout_scaling():
  """Verifies that dropout scales output to maintain expected value."""
  graph = bx.Graph('root')
  rate = 0.5
  dropout = bx.Dropout(graph.child('dropout'), rate=rate)

  # Use large input to get stable statistics.
  x = jnp.ones((1000, 1000))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))

  y, _ = dropout(params, x, is_training=True)

  # Non-zero elements should be scaled by 1/(1-rate) = 2.
  non_zero_mask = y != 0.0
  non_zero_vals = y[non_zero_mask]

  expected_scale = 1.0 / (1.0 - rate)
  assert jnp.allclose(
    non_zero_vals, expected_scale, atol=1e-5
  ), 'Non-zero values should be scaled.'


def test_dropout_zero_rate():
  """Verifies that dropout with rate=0 is identity."""
  graph = bx.Graph('root')
  dropout = bx.Dropout(graph.child('dropout'), rate=0.0)

  x = jnp.ones((10, 10))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))

  y, params_out = dropout(params, x, is_training=True)

  assert jnp.allclose(y, x), 'Dropout with rate=0 should be identity.'
  # Should not consume RNG (counter should not even be initialized).
  counter_path = ('root', 'rng', 'counter')
  assert (
    counter_path not in params_out._data
  ), 'Dropout with rate=0 should not consume RNG.'


def test_dropout_invalid_rate():
  """Verifies that invalid dropout rates raise errors."""
  graph = bx.Graph('root')

  with pytest.raises(ValueError, match='Dropout rate must be in'):
    bx.Dropout(graph.child('dropout'), rate=1.0)

  with pytest.raises(ValueError, match='Dropout rate must be in'):
    bx.Dropout(graph.child('dropout2'), rate=-0.1)
