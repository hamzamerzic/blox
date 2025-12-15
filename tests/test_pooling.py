import jax.numpy as jnp
import blox.blocks as blocks


def test_max_pool_shapes():
  """Verifies max pooling output shapes."""
  # [batch, height, width, channels]
  x = jnp.arange(16).reshape((1, 4, 4, 1)).astype(jnp.float32)

  # 2x2 pooling with stride 2
  y = blocks.max_pool(x, window_shape=2, strides=2)
  assert y.shape == (1, 2, 2, 1)

  # Check values
  # x:
  # 0  1  2  3
  # 4  5  6  7
  # 8  9 10 11
  # 12 13 14 15
  #
  # Top-left 2x2: max(0,1,4,5) = 5
  # Top-right 2x2: max(2,3,6,7) = 7
  # Bot-left 2x2: max(8,9,12,13) = 13
  # Bot-right 2x2: max(10,11,14,15) = 15
  expected = jnp.array([[[[5.0], [7.0]], [[13.0], [15.0]]]])
  assert jnp.allclose(y, expected)


def test_avg_pool_shapes():
  """Verifies average pooling output shapes."""
  # [batch, height, width, channels]
  x = jnp.ones((1, 4, 4, 1))

  # 2x2 pooling with stride 2
  y = blocks.avg_pool(x, window_shape=2, strides=2)
  assert y.shape == (1, 2, 2, 1)
  assert jnp.allclose(y, jnp.ones_like(y))


def test_avg_pool_same_padding_correctness():
  """Verifies average pooling with SAME padding ignores padded zeros.

  We expect the average to be computed over *valid* pixels only (excluding padding).
  """
  # 3x3 input, 2x2 window, stride 1, padding SAME
  # x:
  # 1 1 1
  # 1 1 1
  # 1 1 1
  x = jnp.ones((1, 3, 3, 1))

  # blocks.avg_pool implementation
  y_blox = blocks.avg_pool(x, window_shape=2, strides=1, padding='SAME')

  # If we strictly average valid pixels, the result should be all 1s.
  # The current implementation (dividing by window size) produces 0.25 at corners.
  assert jnp.allclose(
    y_blox, 1.0
  ), f'Expected all 1s, but got min value: {jnp.min(y_blox)}'


def test_max_pool_1d():
  """Verifies 1D max pooling."""
  x = jnp.array([[[1.0], [2.0], [3.0], [4.0]]])  # 1, 4, 1
  y = blocks.max_pool(x, window_shape=2, strides=2)
  assert y.shape == (1, 2, 1)
  assert jnp.allclose(y, jnp.array([[[2.0], [4.0]]]))


def test_min_pool_basic():
  """Verifies min pooling correctness."""
  # [batch, height, width, channels]
  x = jnp.arange(16).reshape((1, 4, 4, 1)).astype(jnp.float32)

  # 2x2 pooling with stride 2
  y = blocks.min_pool(x, window_shape=2, strides=2)
  assert y.shape == (1, 2, 2, 1)

  # Check values - should be top-left element of each 2x2 block
  # Top-left 2x2: min(0,1,4,5) = 0
  # Top-right 2x2: min(2,3,6,7) = 2
  # Bot-left 2x2: min(8,9,12,13) = 8
  # Bot-right 2x2: min(10,11,14,15) = 10
  expected = jnp.array([[[[0.0], [2.0]], [[8.0], [10.0]]]])
  assert jnp.allclose(y, expected)


def test_min_pool_padding():
  """Verifies min pooling with padding handles values correctly."""
  # Input with all negative values to ensure padding doesn't interfere incorrectly
  # (though min_pool should ignore padding if implemented right or if padded with +inf)
  # However, standard reduce_window with min pads with init_value (+inf), so it works naturally.
  x = -jnp.ones((1, 3, 3, 1))

  # Min pool with same padding. Padding with +inf is standard for min pooling.
  # So edge values should remain -1.
  y = blocks.min_pool(x, window_shape=2, strides=1, padding='SAME')

  assert jnp.allclose(y, -1.0)


def test_pool_strides_none():
  """Verifies that strides default to window_shape if None."""
  x = jnp.zeros((1, 4, 4, 1))
  # Window 2, Strides None -> Strides 2
  y = blocks.max_pool(x, window_shape=2, strides=None)
  assert y.shape == (1, 2, 2, 1)

  y = blocks.min_pool(x, window_shape=2, strides=None)
  assert y.shape == (1, 2, 2, 1)

  y = blocks.avg_pool(x, window_shape=2, strides=None)
  assert y.shape == (1, 2, 2, 1)
