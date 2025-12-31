"""Tests for vmap compatibility with auto_fold_in_axes."""

import blox as bx
import jax
import jax.numpy as jnp

# =============================================================================
# Basic auto_fold_in_axes behavior
# =============================================================================


def test_auto_fold_in_axes_noop_outside_vmap():
  """auto_fold_in_axes is a no-op outside vmap."""
  graph = bx.Graph('root')
  rng1 = bx.Rng(graph.child('rng1'), auto_fold_in_axes=True)
  rng2 = bx.Rng(graph.child('rng2'), auto_fold_in_axes=False)

  params = rng1.seed(bx.Params(), seed=42)
  params = rng2.seed(params, seed=42)

  # Outside vmap, both should produce the same keys.
  key1, _ = rng1(params)
  key2, _ = rng2(params)

  assert jnp.array_equal(key1, key2)


def test_auto_fold_in_axes_inside_vmap():
  """auto_fold_in_axes produces different keys per batch element inside vmap."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  keys = jax.vmap(get_key, axis_name='batch')(jnp.ones((4, 1)))

  # Each batch element should have different key.
  for i in range(1, 4):
    assert not jnp.array_equal(keys[0], keys[i])


def test_auto_fold_in_axes_disabled_inside_vmap():
  """auto_fold_in_axes=False produces same keys across batch elements."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=False)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  keys = jax.vmap(get_key, axis_name='batch')(jnp.ones((4, 1)))

  # All batch elements should have the same key.
  for i in range(1, 4):
    assert jnp.array_equal(keys[0], keys[i])


def test_auto_fold_in_axes_multiple_axes():
  """auto_fold_in_axes works with multiple nested axes."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  nested = jax.vmap(jax.vmap(get_key, axis_name='inner'), axis_name='outer')
  keys = nested(jnp.ones((2, 3, 1)))

  # All 6 keys should be unique.
  flat = keys.reshape(6, -1)
  for i in range(6):
    for j in range(i + 1, 6):
      assert not jnp.array_equal(flat[i], flat[j])


# =============================================================================
# RNG key generation with auto_fold_in_axes
# =============================================================================


def test_rng_produces_different_keys_per_batch():
  """Rng produces different keys per batch element with auto_fold_in_axes."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  keys = jax.vmap(get_key, axis_name='batch')(jnp.ones((4, 1)))

  for i in range(1, 4):
    assert not jnp.array_equal(keys[0], keys[i])


def test_without_auto_fold_produces_same_keys():
  """Without auto_fold_in_axes, all batch elements get same RNG keys."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=False)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  keys = jax.vmap(get_key, axis_name='batch')(jnp.ones((4, 1)))

  for i in range(1, 4):
    assert jnp.array_equal(keys[0], keys[i])


# =============================================================================
# Nested vmap behavior
# =============================================================================


def test_nested_vmap_unique_keys():
  """Nested vmap with auto_fold_in_axes produces unique keys at each position."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def inner(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  nested = jax.vmap(jax.vmap(inner, axis_name='inner'), axis_name='outer')
  keys = nested(jnp.ones((2, 3, 1)))  # [outer=2, inner=3, features=1]

  # All 6 keys should be unique.
  flat = keys.reshape(6, -1)
  for i in range(6):
    for j in range(i + 1, 6):
      assert not jnp.array_equal(flat[i], flat[j])


# =============================================================================
# Integration tests - realistic usage patterns
# =============================================================================


def test_vmap_init_without_auto_fold_produces_same_params():
  """Integration: vmap model init without auto_fold_in_axes produces same params."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=False)
  linear = bx.Linear(graph.child('linear'), output_size=4, rng=rng)

  def init(x):
    params = rng.seed(bx.Params(), seed=42)
    _, params = linear(params, x)
    return params.locked()

  params_batch = jax.vmap(init, axis_name='batch')(jnp.ones((4, 1, 3)))

  # All batch elements should have same weights.
  kernel = params_batch._data[('root', 'linear', 'kernel')].value
  for i in range(1, 4):
    assert jnp.allclose(kernel[0], kernel[i])


def test_nested_vmap_mlp_apply():
  """Integration: nested vmap for applying model with different dropout masks."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)
  # Use larger hidden size to avoid statistical collisions.
  layer1 = bx.Linear(graph.child('layer1'), output_size=64, rng=rng)
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)
  layer2 = bx.Linear(graph.child('layer2'), output_size=32, rng=rng)

  # Initialize params once (outside vmap).
  def init_model(x):
    params = rng.seed(bx.Params(), seed=0)
    x, params = layer1(params, x)
    x, params = dropout(params, x, is_training=False)
    _, params = layer2(params, x)
    return params.locked()

  x_sample = jnp.ones((1, 16))
  params = init_model(x_sample)

  # Apply with nested vmap - params replicated, only data and RNG vary.
  def apply_model(params, x):
    # auto_fold_in_axes handles unique dropout masks at each position.
    x, params = layer1(params, x)
    x = jax.nn.relu(x)
    x, params = dropout(params, x, is_training=True)
    x, params = layer2(params, x)
    return x

  # Nested vmap: params replicated (in_axes=None), data batched.
  apply_nested = jax.vmap(
      jax.vmap(apply_model, in_axes=(None, 0), axis_name='inner'),
      in_axes=(None, 0),
      axis_name='outer',
  )

  # Shape: [outer=2, inner=3, batch=1, features=16]
  x = jnp.ones((2, 3, 1, 16))
  out = apply_nested(params, x)

  # All 6 positions should have different outputs due to different dropout.
  flat_out = out.reshape(6, -1)
  for i in range(6):
    for j in range(i + 1, 6):
      assert not jnp.allclose(flat_out[i], flat_out[j])


def test_vmap_dropout_different_masks():
  """Integration: vmap dropout produces different masks per batch element."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

  def apply(x):
    params = rng.seed(bx.Params(), seed=42)
    out, _ = dropout(params, x, is_training=True)
    return out

  x = jnp.ones((4, 16))
  out = jax.vmap(apply, axis_name='batch')(x)

  # Each batch element should have different dropout mask.
  for i in range(1, 4):
    assert not jnp.array_equal(out[0] == 0, out[i] == 0)


def test_vmap_dropout_same_masks_without_auto_fold():
  """Integration: vmap dropout produces same masks without auto_fold_in_axes."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=False)
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

  def apply(x):
    params = rng.seed(bx.Params(), seed=42)
    out, _ = dropout(params, x, is_training=True)
    return out

  x = jnp.ones((4, 16))
  out = jax.vmap(apply, axis_name='batch')(x)

  # All batch elements should have the same dropout mask.
  for i in range(1, 4):
    assert jnp.array_equal(out[0] == 0, out[i] == 0)


# =============================================================================
# Pytree preservation
# =============================================================================


def test_params_preserved_through_jit():
  """Params are preserved through pytree round-trip (jit)."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def check_reconstruction(x):
    params = rng.seed(bx.Params(), seed=42)
    key1, params = rng(params)

    @jax.jit
    def identity(p):
      return p

    params_after = identity(params)
    key2, _ = rng(params_after)
    return key1, key2

  orig, recon = jax.vmap(check_reconstruction, axis_name='batch')(
      jnp.ones((3, 1))
  )
  # Keys should be different because counter incremented.
  # But both should be valid device-unique keys.
  for i in range(1, 3):
    assert not jnp.array_equal(orig[0], orig[i])
    assert not jnp.array_equal(recon[0], recon[i])


def test_rng_counter_increments_correctly():
  """Rng counter increments correctly through multiple calls."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def check_counter(x):
    params = rng.seed(bx.Params(), seed=42)

    counter0 = rng.get_counter(params)
    _, params = rng(params)
    counter1 = rng.get_counter(params)
    _, params = rng(params)
    counter2 = rng.get_counter(params)

    return jnp.array([counter0, counter1, counter2])

  counters = jax.vmap(check_counter, axis_name='batch')(jnp.ones((3, 1)))

  # All batch elements should have same counter progression.
  for i in range(3):
    assert jnp.array_equal(counters[i], jnp.array([0, 1, 2]))


# =============================================================================
# auto_fold_in_axes behavior: init vs runtime
# =============================================================================


def test_vmap_with_axis_name_init_produces_same_params():
  """vmap with axis_name during init produces same params per batch."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)
  linear = bx.Linear(graph.child('linear'), output_size=4, rng=rng)

  def init(x):
    params = rng.seed(bx.Params(), seed=42)
    _, params = linear(params, x)
    key, params = rng(params)
    return key, params.locked()

  # out_axes=(0, None) - keys batched, params must be identical (JAX will fail
  # if params differ across batch elements).
  keys, _ = jax.vmap(init, axis_name='batch', out_axes=(0, None))(
      jnp.ones((4, 1, 3))
  )

  # Keys should be different due to auto_fold_in_axes during runtime.
  for i in range(1, 4):
    assert not jnp.array_equal(keys[0], keys[i])


def test_vmap_without_axis_name_fails():
  """vmap without axis_name raises ValueError with auto_fold_in_axes."""
  import pytest

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

  def apply(x):
    params = rng.seed(bx.Params(), seed=42)
    out, _ = dropout(params, x, is_training=True)
    return out

  x = jnp.ones((4, 16))

  # Unnamed vmap should fail with auto_fold_in_axes=True.
  with pytest.raises(ValueError, match='Unnamed vmap detected'):
    jax.vmap(apply)(x)


def test_unnamed_vmap_works_with_auto_fold_disabled():
  """Unnamed vmap works when auto_fold_in_axes=False."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=False)
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

  def apply(x):
    params = rng.seed(bx.Params(), seed=42)
    out, _ = dropout(params, x, is_training=True)
    return out

  x = jnp.ones((4, 16))
  out = jax.vmap(apply)(x)

  # All batch elements have SAME dropout mask (no axis folding).
  for i in range(1, 4):
    assert jnp.array_equal(out[0], out[i])


def test_unnamed_vmap_works_without_rng_calls():
  """Unnamed vmap works with auto_fold_in_axes=True if no RNG calls made."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)
  linear = bx.Linear(graph.child('linear'), output_size=4, rng=rng)

  # Initialize params outside vmap.
  def init(x):
    params = rng.seed(bx.Params(), seed=42)
    _, params = linear(params, x)
    return params.locked()

  params = init(jnp.ones((1, 3)))

  # Apply without RNG calls (no dropout, just linear).
  def apply(x):
    out, _ = linear(params, x)
    return out

  x = jnp.ones((4, 1, 3))
  # Unnamed vmap works because we don't call rng() during apply.
  out = jax.vmap(apply)(x)

  # All batch elements should have same output (same linear).
  for i in range(1, 4):
    assert jnp.allclose(out[0], out[i])


def test_double_jit_with_nested_vmaps():
  """Double JIT with nested named vmaps produces unique indices."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  @jax.jit
  def inner_jit(data):
    return jax.vmap(get_key, axis_name='v_inner')(data)

  @jax.jit
  def outer_jit(data):
    return jax.vmap(inner_jit, axis_name='v_outer')(data)

  x = jnp.zeros((4, 2, 6))
  keys = outer_jit(x)

  # All 8 keys should be unique.
  flat = keys.reshape(8, -1)
  for i in range(8):
    for j in range(i + 1, 8):
      assert not jnp.array_equal(flat[i], flat[j])


def test_scan_with_vmap():
  """lax.scan with vmap inside produces unique keys per scan step."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  def scan_body(carry, chunk):
    i = carry
    local_keys = jax.vmap(get_key, axis_name='v_inner')(chunk)
    return i + 1, local_keys

  x = jnp.zeros((4, 2, 6))
  _, keys = jax.lax.scan(scan_body, 0, x)

  # Keys should be different per vmap position within each step.
  for step in range(4):
    assert not jnp.array_equal(keys[step, 0], keys[step, 1])


def test_scan_with_jit_vmap():
  """lax.scan with jit(vmap) inside produces unique keys."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  def scan_body(carry, chunk):
    i = carry

    @jax.jit
    def body_step(c):
      return jax.vmap(get_key, axis_name='v_inner')(c)

    local_keys = body_step(chunk)
    return i + 1, local_keys

  x = jnp.zeros((4, 2, 6))
  _, keys = jax.lax.scan(scan_body, 0, x)

  # Keys should be different per vmap position within each step.
  for step in range(4):
    assert not jnp.array_equal(keys[step, 0], keys[step, 1])


def test_unnamed_vmap_inside_jit_fails():
  """Unnamed vmap inside JIT fails with auto_fold_in_axes."""
  import pytest

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  with pytest.raises(ValueError, match='Unnamed vmap detected'):
    jax.jit(jax.vmap(get_key))(jnp.ones((4, 6)))


def test_outer_unnamed_vmap_with_inner_jit_named_vmap():
  """Outer unnamed vmap is invisible to inner jit(named vmap)."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  @jax.jit
  def inner_jit(data):
    return jax.vmap(get_key, axis_name='inner')(data)

  # Outer unnamed vmap is stopped at JIT barrier, so this works.
  x = jnp.zeros((4, 2, 6))
  keys = jax.vmap(inner_jit)(x)

  # Inner vmap produces different keys (only sees 'inner' axis).
  for i in range(4):
    assert not jnp.array_equal(keys[i, 0], keys[i, 1])
