import jax.numpy as jnp
import blox as bx


def test_layernorm_shapes():
  """Verifies LayerNorm output shapes."""
  graph = bx.Graph('root')
  ln = bx.LayerNorm(graph.child('ln'))

  x = jnp.ones((2, 10, 32))
  params = bx.Params(seed=0)

  y, params = ln(params, x)

  assert y.shape == x.shape, 'LayerNorm should preserve shape.'


def test_layernorm_normalization():
  """Verifies that LayerNorm normalizes correctly."""
  graph = bx.Graph('root')
  ln = bx.LayerNorm(graph.child('ln'), use_scale=False, use_bias=False)

  # Create input with known statistics.
  x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
  params = bx.Params(seed=0)

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
  params = bx.Params(seed=0)

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
  params = bx.Params(seed=0)

  y, params = rms(params, x)

  assert y.shape == x.shape, 'RMSNorm should preserve shape.'


def test_rmsnorm_normalization():
  """Verifies that RMSNorm normalizes correctly (no mean subtraction)."""
  graph = bx.Graph('root')
  rms = bx.RMSNorm(graph.child('rms'), use_scale=False)

  x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
  params = bx.Params(seed=0)

  y, _ = rms(params, x)

  # RMS normalization: y = x / sqrt(mean(x^2) + eps)
  expected_rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-5)
  expected = x / expected_rms

  assert jnp.allclose(y, expected, atol=1e-5)
