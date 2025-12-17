import blox.blocks as blocks
import jax.numpy as jnp


def test_max_pool_shapes():
  """Verifies max pooling output shapes."""
  # [batch, height, width, channels]
  x = jnp.arange(16).reshape((1, 4, 4, 1)).astype(jnp.float32)

  # 2x2 pooling with stride 2
  y = blocks.max_pool(x, window_shape=(2, 2), strides=2)
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
  y = blocks.avg_pool(x, window_shape=(2, 2), strides=2)
  assert y.shape == (1, 2, 2, 1)
  assert jnp.allclose(y, jnp.ones_like(y))


def test_avg_pool_same_padding_correctness():
  """Verifies average pooling with SAME padding ignores padded zeros."""
  # 3x3 input, 2x2 window, stride 1, padding SAME
  x = jnp.ones((1, 3, 3, 1))

  y_blox = blocks.avg_pool(x, window_shape=(2, 2), strides=1, padding='SAME')

  # If we strictly average valid pixels, the result should be all 1s.
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
  y = blocks.min_pool(x, window_shape=(2, 2), strides=2)
  assert y.shape == (1, 2, 2, 1)

  # Check values - should be top-left element of each 2x2 block
  expected = jnp.array([[[[0.0], [2.0]], [[8.0], [10.0]]]])
  assert jnp.allclose(y, expected)


def test_min_pool_padding():
  """Verifies min pooling with padding handles values correctly."""
  x = -jnp.ones((1, 3, 3, 1))

  # Min pool with same padding. Padding with +inf is standard for min pooling.
  y = blocks.min_pool(x, window_shape=(2, 2), strides=1, padding='SAME')

  assert jnp.allclose(y, -1.0)


def test_pool_strides_none():
  """Verifies that strides default to window_shape if None."""
  x = jnp.zeros((1, 4, 4, 1))
  # Window (2,2), Strides None -> Strides (2,2)
  y = blocks.max_pool(x, window_shape=(2, 2), strides=None)
  assert y.shape == (1, 2, 2, 1)

  y = blocks.min_pool(x, window_shape=(2, 2), strides=None)
  assert y.shape == (1, 2, 2, 1)

  y = blocks.avg_pool(x, window_shape=(2, 2), strides=None)
  assert y.shape == (1, 2, 2, 1)


def test_max_pool_unbatched():
  """Verifies max pooling works without batch dimension."""
  # No batch dimension: (height, width, channels)
  x = jnp.arange(16).reshape((4, 4, 1)).astype(jnp.float32)

  y = blocks.max_pool(x, window_shape=(2, 2), strides=2)
  assert y.shape == (2, 2, 1)

  expected = jnp.array([[[5.0], [7.0]], [[13.0], [15.0]]])
  assert jnp.allclose(y, expected)


def test_avg_pool_unbatched():
  """Verifies avg pooling works without batch dimension."""
  x = jnp.ones((4, 4, 1))

  y = blocks.avg_pool(x, window_shape=(2, 2), strides=2)
  assert y.shape == (2, 2, 1)
  assert jnp.allclose(y, jnp.ones_like(y))


def test_pool_batched_unbatched_equivalence():
  """Verifies unbatched output matches batched output for batch_size=1."""
  x_unbatched = jnp.arange(16).reshape((4, 4, 1)).astype(jnp.float32)
  x_batched = x_unbatched[None, ...]

  y_unbatched = blocks.max_pool(x_unbatched, window_shape=(2, 2), strides=2)
  y_batched = blocks.max_pool(x_batched, window_shape=(2, 2), strides=2)

  assert y_unbatched.shape == (2, 2, 1)
  assert y_batched.shape == (1, 2, 2, 1)
  assert jnp.allclose(y_unbatched, y_batched[0])


def test_pool_multiple_batch_dims():
  """Verifies pooling works with multiple batch dimensions."""
  # Multiple batch dims: (batch1, batch2, height, width, channels)
  x = jnp.ones((2, 3, 4, 4, 1))

  y = blocks.max_pool(x, window_shape=(2, 2), strides=2)
  assert y.shape == (2, 3, 2, 2, 1)

  y = blocks.avg_pool(x, window_shape=(2, 2), strides=2)
  assert y.shape == (2, 3, 2, 2, 1)


def test_max_pool_explicit_padding():
  """Verifies max pooling with explicit padding pairs."""
  # [batch, height, width, channels]
  x = jnp.ones((1, 4, 4, 1))

  # Pad 1 on each side
  y = blocks.max_pool(
      x, window_shape=(2, 2), strides=2, padding=((1, 1), (1, 1))
  )
  assert y.shape == (1, 3, 3, 1)


def test_avg_pool_explicit_padding():
  """Verifies avg pooling with explicit padding pairs."""
  x = jnp.ones((1, 4, 4, 1))

  # Pad 1 on each side
  y = blocks.avg_pool(
      x, window_shape=(2, 2), strides=2, padding=((1, 1), (1, 1))
  )
  assert y.shape == (1, 3, 3, 1)
  # Average should still be 1.0 since we count valid elements
  assert jnp.allclose(y, 1.0)
