import jax
import jax.numpy as jnp
import blox as bx
import pytest


def test_conv1d_shapes():
  """Verifies 1D convolution output shapes."""
  graph = bx.Graph('root')
  conv = bx.Conv(graph.child('conv'), output_channels=16, kernel_size=3)

  # [batch, length, channels]
  x = jnp.ones((2, 10, 8))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, params = conv(params, x)

  # SAME padding preserves spatial dimensions.
  assert y.shape == (2, 10, 16)


def test_conv2d_shapes():
  """Verifies 2D convolution output shapes."""
  graph = bx.Graph('root')
  conv = bx.Conv(graph.child('conv'), output_channels=32, kernel_size=(3, 3))

  # [batch, height, width, channels]
  x = jnp.ones((2, 28, 28, 3))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, params = conv(params, x)

  # SAME padding preserves spatial dimensions.
  assert y.shape == (2, 28, 28, 32)


def test_conv_valid_padding():
  """Verifies VALID padding reduces spatial dimensions."""
  graph = bx.Graph('root')
  conv = bx.Conv(
    graph.child('conv'), output_channels=16, kernel_size=(3, 3), padding='VALID'
  )

  x = jnp.ones((2, 28, 28, 8))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, _ = conv(params, x)

  # VALID padding: output_size = input_size - kernel_size + 1
  assert y.shape == (2, 26, 26, 16)


def test_conv_strides():
  """Verifies strided convolution reduces spatial dimensions."""
  graph = bx.Graph('root')
  conv = bx.Conv(
    graph.child('conv'), output_channels=16, kernel_size=(3, 3), strides=2
  )

  x = jnp.ones((2, 28, 28, 8))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, _ = conv(params, x)

  # Stride 2 with SAME padding: output_size = ceil(input_size / stride)
  assert y.shape == (2, 14, 14, 16)


def test_conv_dilations():
  """Verifies dilated convolution (atrous convolution)."""
  graph = bx.Graph('root')
  conv = bx.Conv(
    graph.child('conv'),
    output_channels=16,
    kernel_size=(3, 3),
    kernel_dilation=2,
    padding='VALID',
  )

  x = jnp.ones((2, 28, 28, 8))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, _ = conv(params, x)

  # Effective kernel size = 3 + (3-1) * (2-1) = 5
  # Output size = 28 - 5 + 1 = 24
  assert y.shape == (2, 24, 24, 16)


def test_conv_no_bias():
  """Verifies convolution without bias."""
  graph = bx.Graph('root')
  conv = bx.Conv(
    graph.child('conv'), output_channels=16, kernel_size=(3, 3), use_bias=False
  )

  x = jnp.ones((2, 8, 8, 4))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  _, params = conv(params, x)
  params = params.finalize()

  assert ('root', 'conv', 'kernel') in params._data
  assert ('root', 'conv', 'bias') not in params._data


def test_conv_kernel_shape():
  """Verifies kernel shape is correct."""
  graph = bx.Graph('root')
  conv = bx.Conv(graph.child('conv'), output_channels=32, kernel_size=(5, 5))

  x = jnp.ones((2, 16, 16, 8))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  _, params = conv(params, x)
  params = params.finalize()

  # Kernel shape: (kernel_h, kernel_w, in_features, out_features)
  kernel_shape = params._data[('root', 'conv', 'kernel')].value.shape
  assert kernel_shape == (5, 5, 8, 32)


def test_conv_learning():
  """Verifies gradients propagate through convolution.

  Task: Learn a 3x3 kernel that maps constant input to constant output.
  With VALID padding and constant input, every output pixel sees the same
  9 input values, so the optimal kernel has values summing to the target.
  """
  graph = bx.Graph('root')
  conv = bx.Conv(
    graph.child('conv'),
    output_channels=1,
    kernel_size=(3, 3),
    use_bias=False,
    padding='VALID',
  )

  # Constant input of ones, large enough for VALID padding.
  x = jnp.ones((1, 8, 8, 1))
  # Target: constant 5.0 for all output pixels (6x6 with VALID padding).
  target = jnp.ones((1, 6, 6, 1)) * 5.0
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))

  # Initialize.
  _, params = conv(params, x)
  params = params.finalize()

  @jax.jit
  def step(p):
    trainable, non_trainable = p.split()

    def loss(t):
      full_params = t.merge(non_trainable)
      pred, _ = conv(full_params, x)
      return jnp.mean((pred - target) ** 2)

    grads = jax.grad(loss)(trainable)
    new_trainable = jax.tree.map(lambda w, g: w - 0.1 * g, trainable, grads)
    return new_trainable.merge(non_trainable)

  # Train.
  curr = params
  for _ in range(50):
    curr = step(curr)

  pred, _ = conv(curr, x)
  assert jnp.allclose(pred, target, atol=0.01), 'Conv should learn.'


def test_conv_invalid_rank():
  """Verifies error on mismatched input rank."""
  graph = bx.Graph('root')
  conv = bx.Conv(graph.child('conv'), output_channels=16, kernel_size=(3, 3))

  # 1D input for 2D conv.
  x = jnp.ones((2, 10, 8))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  with pytest.raises(ValueError, match='Expected input rank'):
    conv(params, x)


def test_conv3d_shapes():
  """Verifies 3D convolution output shapes."""
  graph = bx.Graph('root')
  conv = bx.Conv(graph.child('conv'), output_channels=16, kernel_size=(3, 3, 3))

  # [batch, depth, height, width, channels]
  x = jnp.ones((2, 8, 8, 8, 4))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, _ = conv(params, x)

  assert y.shape == (2, 8, 8, 8, 16)


def test_conv_depthwise():
  """Verifies depthwise convolution (feature_group_count = input_channels)."""
  graph = bx.Graph('root')
  input_channels = 8
  conv = bx.Conv(
    graph.child('conv'),
    output_channels=8,  # Must equal input_channels for depthwise.
    kernel_size=(3, 3),
    feature_group_count=8,  # One filter per input channel.
  )

  x = jnp.ones((2, 16, 16, input_channels))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, params = conv(params, x)
  params = params.finalize()

  assert y.shape == (2, 16, 16, 8)

  # Kernel shape: (3, 3, 1, 8) - one filter per channel.
  kernel_shape = params._data[('root', 'conv', 'kernel')].value.shape
  assert kernel_shape == (3, 3, 1, 8)
