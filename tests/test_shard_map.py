"""Tests for shard_map compatibility with sharded models.

Without manual RNG folding, all devices get the SAME random keys.
This is expected JAX behavior - users must manually fold in axis indices
to get unique randomness per device.
"""

import blox as bx
import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# Set up fake CPU devices for testing.
chex.set_n_cpu_devices(8)


def get_partition_spec(params: bx.Params):
  """Extracts PartitionSpecs from a Params container."""

  def to_pspec(param):
    if isinstance(param, bx.Param):
      sharding = param.sharding
      if sharding is not None:
        return P(*sharding)
      return P()
    return param

  return jax.tree.map(
      to_pspec, params, is_leaf=lambda x: isinstance(x, bx.Param)
  )


def get_named_sharding(mesh, params: bx.Params):
  """Converts Params metadata to NamedSharding for device_put."""
  specs = get_partition_spec(params)

  def to_sharding(spec):
    if isinstance(spec, P):
      return NamedSharding(mesh, spec)
    return spec

  return jax.tree.map(to_sharding, specs)


# =============================================================================
# End-to-end sharded model: init and apply
# =============================================================================


def test_sharded_linear_model_parallel():
  """End-to-end test: model-parallel Linear with sharded weights.

  With model parallelism, weights are sharded across devices. We init on host
  and use device_put to distribute, then apply inside shard_map.
  """
  mesh = jax.make_mesh((4,), ('model',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  # Weight sharded across output dim (model parallelism).
  linear = bx.Linear(
      graph.child('linear'),
      output_size=16,
      rng=rng,
      kernel_metadata={'sharding': (None, 'model')},
      bias_metadata={'sharding': ('model',)},
  )

  # Initialize on host.
  x_sample = jnp.ones((1, 8))
  params = rng.seed(bx.Params(), seed=42)
  _, params = linear(params, x_sample)
  params = params.locked()

  # Shard params according to metadata.
  param_specs = get_partition_spec(params)
  shardings = get_named_sharding(mesh, params)
  sharded_params = jax.device_put(params, shardings)

  # Apply: each device computes its shard of the output.
  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=(param_specs, P()),
      out_specs=(P(None, 'model'), param_specs),
  )
  def apply_sharded(params, x):
    out, params = linear(params, x)
    return out, params

  x = jnp.ones((2, 8))
  out, out_params = apply_sharded(sharded_params, x)
  assert out.shape == (2, 16)

  # === Verification ===
  expected, _ = linear(params, x)
  assert jnp.allclose(out, expected)


def test_sharded_linear_data_parallel():
  """End-to-end test: data-parallel Linear with replicated params."""
  mesh = jax.make_mesh((4,), ('batch',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  linear = bx.Linear(
      graph.child('linear'),
      output_size=4,
      rng=rng,
      kernel_metadata={'sharding': ()},
      bias_metadata={'sharding': ()},
  )

  # Initialize on host.
  x_sample = jnp.ones((1, 3))
  params = rng.seed(bx.Params(), seed=42)
  _, params = linear(params, x_sample)
  params = params.locked()

  # Replicate params.
  param_specs = get_partition_spec(params)
  shardings = get_named_sharding(mesh, params)
  sharded_params = jax.device_put(params, shardings)

  # Apply with data parallelism.
  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=(param_specs, P('batch', None)),
      out_specs=(P('batch', None), param_specs),
  )
  def apply_data_parallel(params, x):
    out, params = linear(params, x)
    return out, params

  x = jax.device_put(jnp.ones((8, 3)), NamedSharding(mesh, P('batch', None)))
  out, out_params = apply_data_parallel(sharded_params, x)
  assert out.shape == (8, 4)

  # === Verification ===
  expected, _ = linear(params, x)
  assert jnp.allclose(out, expected)


def test_sharded_mlp_tensor_parallel():
  """End-to-end test: tensor-parallel MLP (column then row parallel)."""
  mesh = jax.make_mesh((4,), ('model',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  # Column parallel: shard output.
  layer1 = bx.Linear(
      graph.child('layer1'),
      output_size=16,
      rng=rng,
      kernel_metadata={'sharding': (None, 'model')},
      bias_metadata={'sharding': ('model',)},
  )
  # Row parallel: shard input.
  layer2 = bx.Linear(
      graph.child('layer2'),
      output_size=4,
      rng=rng,
      kernel_metadata={'sharding': ('model', None)},
      bias_metadata={'sharding': ()},
  )

  def mlp(params, x):
    x, params = layer1(params, x)
    x = jax.nn.relu(x)
    x, params = layer2(params, x)
    return x, params

  # Initialize on host.
  x_sample = jnp.ones((1, 8))
  params = rng.seed(bx.Params(), seed=42)
  _, params = mlp(params, x_sample)
  params = params.locked()

  # Shard params.
  param_specs = get_partition_spec(params)
  shardings = get_named_sharding(mesh, params)
  sharded_params = jax.device_put(params, shardings)

  # Apply with tensor parallelism.
  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=(param_specs, P()),
      out_specs=(P(), param_specs),
  )
  def apply_mlp(params, x):
    h, params = layer1(params, x)
    h = jax.nn.relu(h)
    out, params = layer2(params, h)
    out = jax.lax.psum(out, 'model')
    return out, params

  x = jnp.ones((2, 8))
  out, out_params = apply_mlp(sharded_params, x)
  assert out.shape == (2, 4)

  # === Verification ===
  expected, _ = mlp(params, x)
  assert jnp.allclose(out, expected)


# =============================================================================
# Init behavior: same params without manual folding
# =============================================================================


def test_shard_map_init_produces_same_params():
  """Init inside shard_map produces same params across devices.

  Without manual axis folding, all devices get identical params. This is
  expected: for sharded models, use jit for initialization (which handles
  RNG partitioning automatically) rather than shard_map.
  """
  mesh = jax.make_mesh((4,), ('model',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  linear = bx.Linear(
      graph.child('linear'),
      output_size=4,
      rng=rng,
      kernel_metadata={'sharding': ('model',)},
      bias_metadata={'sharding': ('model',)},
  )

  def init_model(x):
    params = rng.seed(bx.Params(), seed=42)
    _, params = linear(params, x)
    return params.locked()

  x_sample = jnp.ones((1, 3))

  # Get structure.
  params_structure = jax.eval_shape(init_model, x_sample)
  param_specs = get_partition_spec(params_structure)

  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=P(),
      out_specs=param_specs,
  )
  def init_sharded(x):
    return init_model(x)

  params = init_sharded(x_sample)

  # === Verification ===
  # All devices should have identical weights (no manual folding).
  kernel_shards = params._data[
      ('root', 'linear', 'kernel')
  ].value.addressable_shards
  kernel_shard_data = [np.asarray(s.data) for s in kernel_shards]
  assert np.allclose(
      kernel_shard_data[0], kernel_shard_data[1]
  ), 'Params should be identical across devices without manual folding'


# =============================================================================
# LayerNorm with cross-device aggregation
# =============================================================================


def test_layernorm_cross_device():
  """Test LayerNorm with axis_name aggregates mean/var across devices."""
  mesh = jax.make_mesh((4,), ('batch',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  ln = bx.LayerNorm(
      graph.child('ln'), use_scale=False, use_bias=False, axis_name='batch'
  )

  # Each device has different samples.
  x = jnp.concatenate(
      [
          jnp.tile(jnp.array([[i, i + 0.5, i + 1, i + 1.5]]), (2, 1))
          for i in range(4)
      ],
      axis=0,
  )  # Shape (8, 4)

  params = rng.seed(bx.Params(), seed=0)

  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=(P(), P('batch', None)),
      out_specs=P('batch', None),
  )
  def apply_ln(params, x):
    out, _ = ln(params, x)
    return out

  x = jax.device_put(x, NamedSharding(mesh, P('batch', None)))
  out = apply_ln(params, x)

  # === Verification ===
  global_mean = 2.25
  global_var = 0.3125
  expected = (x - global_mean) / jnp.sqrt(global_var + 1e-5)
  assert jnp.allclose(out, expected, atol=1e-5)


# =============================================================================
# RNG behavior: same keys without manual folding
# =============================================================================


def test_dropout_same_masks_without_manual_folding():
  """Dropout produces same mask on all devices without manual folding."""
  mesh = jax.make_mesh((4,), ('batch',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=P('batch', None),
      out_specs=P('batch', None),
  )
  def apply_dropout(x):
    params = rng.seed(bx.Params(), seed=42)
    out, _ = dropout(params, x, is_training=True)
    return out

  x = jax.device_put(
      jnp.ones((8, 16)), NamedSharding(mesh, P('batch', None))
  )
  out = apply_dropout(x)

  # === Verification ===
  # All devices have SAME dropout mask (no axis folding).
  out_np = np.asarray(out).reshape(4, 2, 16)
  for i in range(1, 4):
    np.testing.assert_allclose(
        out_np[0], out_np[i]
    ), 'Without manual folding, all devices should have same dropout mask'


# =============================================================================
# Manual RNG folding pattern for shard_map
# =============================================================================


def test_manual_folding_produces_different_masks():
  """Manual axis folding produces different dropout masks per device."""
  mesh = jax.make_mesh((4,), ('batch',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=P('batch', None),
      out_specs=P('batch', None),
  )
  def apply_dropout_with_folding(x):
    params = rng.seed(bx.Params(), seed=42)

    # Manual folding pattern:
    # 1. Get the original seed
    original_seed = rng.get_seed(params)

    # 2. Fold in the axis index to get a unique seed per device
    folded_seed = jax.random.fold_in(original_seed, jax.lax.axis_index('batch'))

    # 3. Update params with the folded seed
    params = rng.seed(params, seed=folded_seed)

    # 4. Now dropout produces unique masks per device
    out, _ = dropout(params, x, is_training=True)
    return out

  x = jax.device_put(
      jnp.ones((8, 16)), NamedSharding(mesh, P('batch', None))
  )
  out = apply_dropout_with_folding(x)

  # === Verification ===
  # Different devices should have DIFFERENT dropout masks.
  out_np = np.asarray(out).reshape(4, 2, 16)
  zeros_per_device = [int(np.sum(out_np[i] == 0.0)) for i in range(4)]
  assert not all(
      z == zeros_per_device[0] for z in zeros_per_device
  ), 'With manual folding, different devices should have different masks'


def test_manual_folding_in_shard_map_with_vmap():
  """Manual folding works with shard_map + nested vmap."""
  mesh = jax.make_mesh((2,), ('x',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))

  def get_key_with_folding(x):
    params = rng.seed(bx.Params(), seed=42)

    # Fold in both shard and vmap axes.
    original_seed = rng.get_seed(params)
    shard_idx = jax.lax.axis_index('x')
    vmap_idx = jax.lax.axis_index('v_inner')

    folded_seed = jax.random.fold_in(original_seed, shard_idx)
    folded_seed = jax.random.fold_in(folded_seed, vmap_idx)

    params = rng.seed(params, seed=folded_seed)
    key, _ = rng(params)
    return key

  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=P('x'),
      out_specs=P('x'),
  )
  def sharded_fn(y):
    return jax.vmap(get_key_with_folding, axis_name='v_inner')(y[0])[None]

  x = jax.device_put(jnp.zeros((2, 3, 6)), NamedSharding(mesh, P('x', None, None)))
  keys = sharded_fn(x)

  # All 6 keys should be unique (2 shards * 3 vmap positions).
  keys_np = np.asarray(jax.random.key_data(keys)).reshape(6, -1)
  for i in range(6):
    for j in range(i + 1, 6):
      assert not np.array_equal(keys_np[i], keys_np[j])
