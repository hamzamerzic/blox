import jax.numpy as jnp
import blox as bx


def test_layernorm_shapes():
  """Verifies LayerNorm output shapes."""
  graph = bx.Graph('root')
  ln = bx.LayerNorm(graph.child('ln'))

  x = jnp.ones((2, 10, 32))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, params = ln(params, x)

  assert y.shape == x.shape, 'LayerNorm should preserve shape.'


def test_layernorm_normalization():
  """Verifies that LayerNorm normalizes correctly."""
  graph = bx.Graph('root')
  ln = bx.LayerNorm(graph.child('ln'), use_scale=False, use_bias=False)

  # Create input with known statistics.
  x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, _ = ln(params, x)

  # Output should have mean ~0 and std ~1 along last axis.
  mean = jnp.mean(y, axis=-1)
  std = jnp.std(y, axis=-1)

  assert jnp.allclose(mean, 0.0, atol=1e-5), 'Mean should be ~0.'
  assert jnp.allclose(std, 1.0, atol=1e-5), 'Std should be ~1.'


def test_layernorm_learnable_params():
  """Verifies LayerNorm creates scale and bias parameters."""
  graph = bx.Graph('root')
  ln = bx.LayerNorm(graph.child('ln'))

  x = jnp.ones((2, 16))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  _, params = ln(params, x)
  params = params.finalize()

  assert ('root', 'ln', 'scale') in params._data
  assert ('root', 'ln', 'bias') in params._data
  assert params._data[('root', 'ln', 'scale')].value.shape == (16,)
  assert params._data[('root', 'ln', 'bias')].value.shape == (16,)


def test_rmsnorm_shapes():
  """Verifies RMSNorm output shapes."""
  graph = bx.Graph('root')
  rms = bx.RMSNorm(graph.child('rms'))

  x = jnp.ones((2, 10, 32))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, params = rms(params, x)

  assert y.shape == x.shape, 'RMSNorm should preserve shape.'


def test_rmsnorm_normalization():
  """Verifies that RMSNorm normalizes correctly (no mean subtraction)."""
  graph = bx.Graph('root')
  rms = bx.RMSNorm(graph.child('rms'), use_scale=False)

  x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, _ = rms(params, x)

  # RMS normalization: y = x / sqrt(mean(x^2) + eps)
  expected_rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-5)
  expected = x / expected_rms

  assert jnp.allclose(y, expected, atol=1e-5)


def test_batchnorm_shapes():
  """Verifies BatchNorm output shapes."""
  graph = bx.Graph('root')
  bn = bx.BatchNorm(graph.child('bn'))

  x = jnp.ones((4, 8, 8, 32))  # NHWC format
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, params = bn(params, x, is_training=True)

  assert y.shape == x.shape, 'BatchNorm should preserve shape.'


def test_batchnorm_training_vs_inference():
  """Verifies BatchNorm uses batch vs running stats correctly."""
  graph = bx.Graph('root')
  bn = bx.BatchNorm(graph.child('bn'), use_scale=False, use_bias=False)

  # Create batches with different statistics.
  batch1 = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
  batch2 = jnp.array([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]])

  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  # Training: uses batch statistics.
  y1, params = bn(params, batch1, is_training=True)
  params = params.finalize()

  # The output should be normalized using batch1's statistics.
  mean1 = jnp.mean(batch1, axis=0)
  var1 = jnp.var(batch1, axis=0)
  expected1 = (batch1 - mean1) / jnp.sqrt(var1 + 1e-5)
  assert jnp.allclose(y1, expected1, atol=1e-5)

  # Inference: uses running statistics (which should be close to batch1's stats).
  y2, _ = bn(params, batch2, is_training=False)

  # Output should NOT be normalized using batch2's statistics.
  mean2 = jnp.mean(batch2, axis=0)
  var2 = jnp.var(batch2, axis=0)
  expected_with_batch_stats = (batch2 - mean2) / jnp.sqrt(var2 + 1e-5)

  # y2 should differ from what we'd get with batch statistics.
  assert not jnp.allclose(y2, expected_with_batch_stats, atol=1e-3)


def test_batchnorm_running_stats_update():
  """Verifies running statistics are updated during training."""
  graph = bx.Graph('root')
  bn = bx.BatchNorm(graph.child('bn'), momentum=0.1)

  # Use data with non-zero mean and non-unit variance.
  x = jnp.array([[1.0, 10.0], [5.0, 20.0]])
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  # Initialize.
  _, params = bn(params, x, is_training=True)
  params = params.finalize()

  # Get initial running stats (after first update from init zeros/ones).
  initial_mean = params._data[('root', 'bn', 'running_mean')].value
  initial_var = params._data[('root', 'bn', 'running_var')].value

  # Run another training step - running stats should update.
  _, params = bn(params, x, is_training=True)

  new_mean = params._data[('root', 'bn', 'running_mean')].value
  new_var = params._data[('root', 'bn', 'running_var')].value

  # Running stats should have changed (converging toward batch statistics).
  # EMA formula: new = momentum * old + (1 - momentum) * batch
  # With momentum=0.1: new = 0.1 * old + 0.9 * batch
  batch_mean = jnp.mean(x, axis=0)  # [3.0, 15.0]
  batch_var = jnp.var(x, axis=0)  # [4.0, 25.0]

  # The new stats should follow the EMA update formula.
  expected_mean = 0.1 * initial_mean + 0.9 * batch_mean
  expected_var = 0.1 * initial_var + 0.9 * batch_var
  assert jnp.allclose(new_mean, expected_mean, atol=1e-5)
  assert jnp.allclose(new_var, expected_var, atol=1e-5)


def test_batchnorm_learnable_params():
  """Verifies BatchNorm creates scale and bias parameters."""
  graph = bx.Graph('root')
  bn = bx.BatchNorm(graph.child('bn'))

  x = jnp.ones((2, 16))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  _, params = bn(params, x, is_training=True)
  params = params.finalize()

  assert ('root', 'bn', 'scale') in params._data
  assert ('root', 'bn', 'bias') in params._data
  assert ('root', 'bn', 'running_mean') in params._data
  assert ('root', 'bn', 'running_var') in params._data

  assert params._data[('root', 'bn', 'scale')].value.shape == (16,)
  assert params._data[('root', 'bn', 'bias')].value.shape == (16,)
  assert params._data[('root', 'bn', 'running_mean')].trainable is False
  assert params._data[('root', 'bn', 'running_var')].trainable is False
