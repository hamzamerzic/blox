"""Tests for vmap compatibility with fold_in_axes/fold_out_axes."""

import jax
import jax.numpy as jnp
import blox as bx


# =============================================================================
# Basic fold_in_axes behavior
# =============================================================================


def test_fold_in_axes_noop_outside_transformation():
  """fold_in_axes is a no-op outside vmap/shard_map."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)
  params = bx.Params(rng=rng)
  params_folded = params.fold_in_axes('batch')

  assert params_folded is params
  assert params.folded_axes == ()


def test_fold_in_axes_sets_folded_axes_inside_vmap():
  """fold_in_axes sets folded_axes inside vmap."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def check_folding(x):
    p = bx.Params(rng=rng)
    assert p.folded_axes == ()

    p = p.fold_in_axes('batch')
    assert p.folded_axes == ('batch',)

    # Generate a key to verify folding works.
    key, _ = p.next_key()
    return key

  keys = jax.vmap(check_folding, axis_name='batch')(jnp.ones((4, 1)))

  # Each batch element should have different folded key.
  for i in range(1, 4):
    assert not jnp.array_equal(keys[0], keys[i])


def test_fold_in_axes_idempotent():
  """Folding the same axis twice is a no-op."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def check_idempotent(x):
    p = bx.Params(rng=rng).fold_in_axes('batch')
    p2 = p.fold_in_axes('batch')
    return p is p2

  results = jax.vmap(check_idempotent, axis_name='batch')(jnp.ones((2, 1)))
  assert all(results)


def test_fold_in_axes_multiple():
  """fold_in_axes can fold multiple axes at once."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def check_multi(x):
    p = bx.Params(rng=rng).fold_in_axes('outer', 'inner')
    assert p.folded_axes == ('outer', 'inner')
    key, _ = p.next_key()
    return key

  nested = jax.vmap(jax.vmap(check_multi, axis_name='inner'), axis_name='outer')
  keys = nested(jnp.ones((2, 3, 1)))
  assert keys.shape[0] == 2
  assert keys.shape[1] == 3


# =============================================================================
# fold_out_axes behavior
# =============================================================================


def test_fold_out_axes_noop_outside_transformation():
  """fold_out_axes is a no-op outside vmap/shard_map."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)
  params = bx.Params(rng=rng)
  params_unfolded = params.fold_out_axes('batch')

  assert params_unfolded is params


def test_fold_out_axes_removes_axis():
  """fold_out_axes removes axis from folded_axes."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def check_unfold(x):
    p = bx.Params(rng=rng).fold_in_axes('batch')
    assert p.folded_axes == ('batch',)

    p = p.fold_out_axes('batch')
    assert p.folded_axes == ()
    return True

  results = jax.vmap(check_unfold, axis_name='batch')(jnp.ones((2, 1)))
  assert all(results)


def test_fold_out_axes_multiple():
  """fold_out_axes can unfold multiple axes at once."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def check_multi_unfold(x):
    p = bx.Params(rng=rng).fold_in_axes('outer', 'inner')
    assert p.folded_axes == ('outer', 'inner')

    # Order doesn't matter as long as they're at the tail.
    p = p.fold_out_axes('inner', 'outer')
    assert p.folded_axes == ()
    return True

  nested = jax.vmap(
    jax.vmap(check_multi_unfold, axis_name='inner'), axis_name='outer'
  )
  results = nested(jnp.ones((2, 3, 1)))
  assert jnp.all(results)


def test_fold_out_axes_partial():
  """fold_out_axes can unfold only some axes."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def check_partial(x):
    p = bx.Params(rng=rng).fold_in_axes('outer', 'inner')
    key_both, _ = p.next_key()

    # Unfold only inner.
    p2 = p.fold_out_axes('inner')
    key_outer, _ = p2.next_key()

    # Encode folded_axes check as int: 1 if correct, 0 otherwise.
    axes_correct = jnp.array(p2.folded_axes == ('outer',), dtype=jnp.int32)

    return key_both, key_outer, axes_correct

  nested = jax.vmap(
    jax.vmap(check_partial, axis_name='inner'), axis_name='outer'
  )
  keys_both, keys_outer, axes_correct = nested(jnp.ones((2, 3, 1)))

  # After unfolding inner, folded_axes should be ('outer',).
  assert jnp.all(axes_correct == 1)

  # Inner dimension values should all be same within an outer (since inner is unfolded).
  for outer_idx in range(2):
    for inner_idx in range(1, 3):
      assert jnp.array_equal(
        keys_outer[outer_idx, 0], keys_outer[outer_idx, inner_idx]
      )


# =============================================================================
# RNG key generation with fold_in_axes
# =============================================================================


def test_fold_in_axes_produces_different_keys():
  """fold_in_axes produces different RNG keys per batch element."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def get_key(x):
    params = bx.Params(rng=rng).fold_in_axes('batch')
    key, _ = params.next_key()
    return key

  keys = jax.vmap(get_key, axis_name='batch')(jnp.ones((4, 1)))

  for i in range(1, 4):
    assert not jnp.array_equal(keys[0], keys[i])


def test_without_fold_in_axes_produces_same_keys():
  """Without fold_in_axes, all batch elements get same RNG keys."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def get_key(x):
    params = bx.Params(rng=rng)
    key, _ = params.next_key()
    return key

  keys = jax.vmap(get_key, axis_name='batch')(jnp.ones((4, 1)))

  for i in range(1, 4):
    assert jnp.array_equal(keys[0], keys[i])


# =============================================================================
# Pytree round-trip behavior
# =============================================================================


def test_folded_axes_preserved_after_jit():
  """folded_axes is preserved after pytree round-trip (jit)."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def check_reconstruction(x):
    params = bx.Params(rng=rng).fold_in_axes('batch')
    original_key, _ = params.next_key()

    @jax.jit
    def identity(p):
      return p

    params_after = identity(params)
    reconstructed_key, _ = params_after.next_key()
    return original_key, reconstructed_key

  orig, recon = jax.vmap(check_reconstruction, axis_name='batch')(
    jnp.ones((3, 1))
  )
  # Keys should be different because counter incremented.
  # But both should be valid device-unique keys.
  for i in range(1, 3):
    assert not jnp.array_equal(orig[0], orig[i])
    assert not jnp.array_equal(recon[0], recon[i])


def test_folded_axes_preserved_in_pytree():
  """folded_axes are preserved through pytree operations."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def check_preserved(x):
    p = bx.Params(rng=rng).fold_in_axes('batch')

    @jax.jit
    def identity(params):
      return params

    p_after = identity(p)
    return p_after.folded_axes == ('batch',)

  results = jax.vmap(check_preserved, axis_name='batch')(jnp.ones((3, 1)))
  assert all(results)


# =============================================================================
# Nested vmap behavior
# =============================================================================


def test_nested_vmap_fold_in_both_axes():
  """Nested vmap with fold_in_axes produces unique keys at each position."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def inner(x):
    p = bx.Params(rng=rng).fold_in_axes('outer', 'inner')
    key, _ = p.next_key()
    return key

  nested = jax.vmap(jax.vmap(inner, axis_name='inner'), axis_name='outer')
  keys = nested(jnp.ones((2, 3, 1)))  # [outer=2, inner=3, features=1]

  # All 6 keys should be unique.
  flat = keys.reshape(6, -1)
  for i in range(6):
    for j in range(i + 1, 6):
      assert not jnp.array_equal(flat[i], flat[j])


def test_fold_outer_then_inner_equals_fold_both_in_inner():
  """Folding outer then inner == folding both in inner function."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def approach1(x):
    """Fold outer in outer, fold inner in inner."""
    params_outer = bx.Params(rng=rng).fold_in_axes('outer')

    def inner(xi):
      p = params_outer.fold_in_axes('inner')
      key, _ = p.next_key()
      return key

    return jax.vmap(inner, axis_name='inner')(x)

  def approach2(x):
    """Fold both axes in inner function."""

    def inner(xi):
      p = bx.Params(rng=rng).fold_in_axes('outer', 'inner')
      key, _ = p.next_key()
      return key

    return jax.vmap(inner, axis_name='inner')(x)

  x = jnp.ones((3, 1))
  keys1 = jax.vmap(approach1, axis_name='outer')(jnp.stack([x, x]))
  keys2 = jax.vmap(approach2, axis_name='outer')(jnp.stack([x, x]))

  assert jnp.allclose(keys1, keys2)


def test_nested_vmap_all_positions_unique():
  """Nested vmap with fold_in_axes produces unique keys at each position."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), seed=42)

  def inner(x):
    p = bx.Params(rng=rng).fold_in_axes('outer', 'inner')
    key, _ = p.next_key()
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


def test_vmap_init_produces_different_params():
  """Integration: vmap model init produces different params per batch."""
  graph = bx.Graph('root')
  linear = bx.Linear(graph.child('linear'), output_size=4)
  rng = bx.Rng(graph.child('rng'), seed=42)

  def init(x):
    params = bx.Params(rng=rng).fold_in_axes('batch')
    _, params = linear(params, x)
    return params.finalize()

  params_batch = jax.vmap(init, axis_name='batch')(jnp.ones((4, 1, 3)))

  # Each batch element should have different weights.
  kernel = params_batch._data[('root', 'linear', 'kernel')].value
  for i in range(1, 4):
    assert not jnp.allclose(kernel[0], kernel[i])


def test_nested_vmap_mlp_apply():
  """Integration: nested vmap for applying model with different dropout masks."""
  graph = bx.Graph('root')
  # Use larger hidden size to avoid statistical collisions.
  layer1 = bx.Linear(graph.child('layer1'), output_size=64)
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5)
  layer2 = bx.Linear(graph.child('layer2'), output_size=32)
  rng = bx.Rng(graph.child('rng'), seed=0)

  # Initialize params once (outside vmap).
  def init_model(x):
    params = bx.Params(rng=rng)
    x, params = layer1(params, x)
    x, params = dropout(params, x, is_training=False)
    _, params = layer2(params, x)
    return params.finalize()

  x_sample = jnp.ones((1, 16))
  params = init_model(x_sample)

  # Apply with nested vmap - params replicated, only data and RNG vary.
  def apply_model(params, x):
    # Fold in both axes for unique dropout masks at each position.
    params = params.fold_in_axes('outer', 'inner')
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
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5)
  rng = bx.Rng(graph.child('rng'), seed=42)

  def apply(x):
    params = bx.Params(rng=rng).fold_in_axes('batch')
    out, _ = dropout(params, x, is_training=True)
    return out

  x = jnp.ones((4, 16))
  out = jax.vmap(apply, axis_name='batch')(x)

  # Each batch element should have different dropout mask.
  for i in range(1, 4):
    assert not jnp.array_equal(out[0] == 0, out[i] == 0)
