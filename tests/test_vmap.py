"""Tests for vmap compatibility.

Without manual RNG folding, all batch elements get the SAME random keys.
This is expected JAX behavior - users must manually fold in axis indices
to get unique randomness per batch element.
"""

import jax
import jax.numpy as jnp

import blox as bx

# =============================================================================
# Basic vmap compatibility
# =============================================================================


def test_vmap_init_produces_same_params():
  """vmap model init produces same params across batch elements."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
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


def test_vmap_apply_without_rng_works():
  """vmap apply (no RNG calls) works correctly."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  linear = bx.Linear(graph.child('linear'), output_size=4, rng=rng)

  # Initialize params outside vmap.
  params = rng.seed(bx.Params(), seed=42)
  _, params = linear(params, jnp.ones((1, 3)))
  params = params.locked()

  # Apply without RNG calls (no dropout, just linear).
  def apply(x):
    out, _ = linear(params, x)
    return out

  x = jnp.ones((4, 1, 3))
  out = jax.vmap(apply)(x)

  # All batch elements should have same output (same linear, same input).
  for i in range(1, 4):
    assert jnp.allclose(out[0], out[i])


def test_rng_counter_increments_correctly_in_vmap():
  """Rng counter increments correctly through multiple calls in vmap."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))

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


def test_params_preserved_through_jit_in_vmap():
  """Params are preserved through pytree round-trip (jit) in vmap."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))

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
  for i in range(3):
    assert not jnp.array_equal(orig[i], recon[i])


# =============================================================================
# RNG behavior: same keys without manual folding
# =============================================================================


def test_vmap_produces_same_keys_without_manual_folding():
  """vmap produces same RNG keys across batch without manual folding.

  This is expected JAX behavior! Without manual axis folding, all batch
  elements get identical random values.
  """
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  keys = jax.vmap(get_key, axis_name='batch')(jnp.ones((4, 1)))

  # All batch elements get the SAME key without manual folding.
  for i in range(1, 4):
    assert jnp.array_equal(keys[0], keys[i])


def test_vmap_dropout_same_masks_without_manual_folding():
  """vmap dropout produces same masks without manual axis folding."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

  def apply(x):
    params = rng.seed(bx.Params(), seed=42)
    out, _ = dropout(params, x, is_training=True)
    return out

  x = jnp.ones((4, 16))
  out = jax.vmap(apply, axis_name='batch')(x)

  # All batch elements have SAME dropout mask (no axis folding).
  for i in range(1, 4):
    assert jnp.array_equal(out[0] == 0, out[i] == 0)


# =============================================================================
# Manual RNG folding pattern
# =============================================================================


def test_explicit_batch_index_produces_different_keys():
  """Explicit batch index folding produces different keys per batch element.

  This is the simplest pattern: pass the batch index as an argument.
  """
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))

  def get_key_with_explicit_index(x, batch_idx):
    params = rng.seed(bx.Params(), seed=42)

    # Fold in the explicit batch index.
    original_seed = rng.get_seed(params)
    folded_seed = jax.random.fold_in(original_seed, batch_idx)
    params = rng.seed(params, seed=folded_seed)

    key, params = rng(params)
    return key

  # Pass jnp.arange(4) as the batch indices.
  batch_indices = jnp.arange(4)
  keys = jax.vmap(get_key_with_explicit_index, in_axes=(0, 0))(
      jnp.ones((4, 1)), batch_indices
  )

  # Each batch element now has a DIFFERENT key.
  for i in range(1, 4):
    assert not jnp.array_equal(keys[0], keys[i])


def test_axis_index_produces_different_keys():
  """jax.lax.axis_index folding produces different keys per batch element.

  This demonstrates using axis_name with vmap to get the index implicitly.
  """
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))

  def get_key_with_axis_index(x):
    params = rng.seed(bx.Params(), seed=42)

    # Use jax.lax.axis_index to get the lane index.
    original_seed = rng.get_seed(params)
    folded_seed = jax.random.fold_in(original_seed, jax.lax.axis_index('batch'))
    params = rng.seed(params, seed=folded_seed)

    key, params = rng(params)
    return key

  # axis_name='batch' is required for jax.lax.axis_index.
  keys = jax.vmap(get_key_with_axis_index, axis_name='batch')(jnp.ones((4, 1)))

  # Each batch element now has a DIFFERENT key.
  for i in range(1, 4):
    assert not jnp.array_equal(keys[0], keys[i])


def test_manual_folding_with_dropout():
  """Manual axis folding gives different dropout masks per batch element."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

  def apply_with_folding(x):
    params = rng.seed(bx.Params(), seed=42)

    # Fold in axis index for unique randomness.
    original_seed = rng.get_seed(params)
    folded_seed = jax.random.fold_in(original_seed, jax.lax.axis_index('batch'))
    params = rng.seed(params, seed=folded_seed)

    out, _ = dropout(params, x, is_training=True)
    return out

  x = jnp.ones((4, 16))
  out = jax.vmap(apply_with_folding, axis_name='batch')(x)

  # Each batch element now has a DIFFERENT dropout mask.
  for i in range(1, 4):
    assert not jnp.array_equal(out[0] == 0, out[i] == 0)


def test_manual_folding_nested_vmap():
  """Manual folding works with nested vmaps."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))

  def get_key_with_folding(x):
    params = rng.seed(bx.Params(), seed=42)

    # Fold in both axes for unique keys across all positions.
    original_seed = rng.get_seed(params)

    # Get combined index from both axes.
    outer_idx = jax.lax.axis_index('outer')
    inner_idx = jax.lax.axis_index('inner')

    # Fold in both indices (order matters but is deterministic).
    folded_seed = jax.random.fold_in(original_seed, outer_idx)
    folded_seed = jax.random.fold_in(folded_seed, inner_idx)

    params = rng.seed(params, seed=folded_seed)
    key, _ = rng(params)
    return key

  nested = jax.vmap(
      jax.vmap(get_key_with_folding, axis_name='inner'),
      axis_name='outer',
  )
  keys = nested(jnp.ones((2, 3, 1)))

  # All 6 keys should be unique.
  flat = keys.reshape(6, -1)
  for i in range(6):
    for j in range(i + 1, 6):
      assert not jnp.array_equal(flat[i], flat[j])


def test_manual_folding_restores_seed_for_replicated_params():
  """Manual folding with seed restoration for replicated output params.

  When params must be identical across lanes (out_axes=None), the seed
  must be restored to the original value before returning. The counter
  is the same across lanes since we run the same function with the same
  number of RNG calls.
  """
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  linear = bx.Linear(graph.child('linear'), output_size=4, rng=rng)

  def init_and_apply_with_folding(x):
    params = rng.seed(bx.Params(), seed=42)
    _, params = linear(params, x)
    params = params.locked()

    # For runtime operations, fold in axis index.
    original_seed = rng.get_seed(params)
    folded_seed = jax.random.fold_in(original_seed, jax.lax.axis_index('batch'))

    # Seed can be modified even on locked params (only structure is locked).
    params = rng.seed(params, seed=folded_seed)

    # Get unique key per lane.
    key, params = rng(params)

    # IMPORTANT: Restore original seed before returning replicated params.
    params = rng.seed(params, seed=original_seed)

    return key, params

  # out_axes=(0, None) means keys are batched but params must be identical.
  keys, params = jax.vmap(
      init_and_apply_with_folding,
      axis_name='batch',
      out_axes=(0, None),
  )(jnp.ones((4, 1, 3)))

  # Keys should be different due to manual folding.
  for i in range(1, 4):
    assert not jnp.array_equal(keys[0], keys[i])

  # Params should have the original seed restored.
  final_seed = rng.get_seed(params)
  expected_seed = jax.random.key(42)
  assert jnp.array_equal(final_seed, expected_seed)


def test_init_vs_runtime_folding_pattern():
  """Demonstrates the init vs runtime folding pattern.

  During initialization (params unlocked), we do NOT fold in axis indices
  because we want identical params across all batch elements.

  During runtime (params locked), we DO fold in axis indices to get
  unique randomness per batch element.

  This pattern allows using the same vmap function for both init and runtime.
  """
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

  def forward(params, x):
    original_seed = rng.get_seed(params)

    # Check if we're in init mode (unlocked) or runtime mode (locked).
    if params.is_locked:
      # Runtime: fold in axis index for unique randomness per lane.
      folded_seed = jax.random.fold_in(
          original_seed, jax.lax.axis_index('batch')
      )
      params = rng.seed(params, seed=folded_seed)

    out, params = dropout(params, x, is_training=True)

    # Restore original seed (no-op during init, required for runtime).
    params = rng.seed(params, seed=original_seed)

    return out, params

  # === Init phase: unlocked params, no folding ===
  def init(x):
    params = rng.seed(bx.Params(), seed=42)
    _, params = forward(params, x)
    return params.locked()

  # Init with vmap - params should be identical across batch.
  params = jax.vmap(init, axis_name='batch', out_axes=None)(
      jnp.ones((4, 1, 16))
  )

  # === Runtime phase: locked params, with folding ===
  x = jnp.ones((4, 1, 16))
  out, _ = jax.vmap(forward, in_axes=(None, 0), axis_name='batch')(params, x)

  # Runtime outputs should differ due to folding (different dropout masks).
  for i in range(1, 4):
    assert not jnp.allclose(out[0], out[i])
