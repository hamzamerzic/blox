"""Tests for shard_map compatibility with sharded models."""

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
  params = params.finalized()

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
  params = params.finalized()

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

  x = jnp.ones((8, 3))
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
  params = params.finalized()

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
# auto_fold_in_axes behavior: init vs runtime
# =============================================================================


def test_shard_map_init_produces_same_params():
  """Init inside shard_map produces same params across devices.

  auto_fold_in_axes is NOT applied during param initialization - all devices
  get identical params. This is intentional: for sharded models, use jit for
  initialization (which handles RNG partitioning automatically) rather than
  shard_map. auto_fold_in_axes only applies at runtime (e.g., dropout).
  """
  mesh = jax.make_mesh((4,), ('model',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)
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
    # Also get a runtime key to verify auto_fold_in_axes works there.
    key, params = rng(params)
    # Add dimension for sharding.
    return key[None], params.finalized()

  x_sample = jnp.ones((1, 3))

  # Get structure.
  _, params_structure = jax.eval_shape(init_model, x_sample)
  param_specs = get_partition_spec(params_structure)

  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=P(),
      out_specs=(P('model'), param_specs),
  )
  def init_sharded(x):
    return init_model(x)

  keys, params = init_sharded(x_sample)

  # === Verification ===
  # Params should be identical across devices.
  kernel_shards = params._data[
      ('root', 'linear', 'kernel')
  ].value.addressable_shards
  kernel_shard_data = [np.asarray(s.data) for s in kernel_shards]
  assert np.allclose(
      kernel_shard_data[0], kernel_shard_data[1]
  ), 'Params should be identical across devices during init'

  # Keys should differ due to auto_fold_in_axes at runtime.
  for i in range(1, 4):
    assert not jnp.array_equal(
        keys[0], keys[i]
    ), 'Runtime keys should differ across devices with auto_fold_in_axes'


def test_init_without_auto_fold_in_axes_produces_same_params():
  """Init inside shard_map WITHOUT auto_fold_in_axes produces identical params.

  When auto_fold_in_axes=False, all devices use the same RNG sequence,
  resulting in identical parameters (which is usually wrong for sharded init).
  """
  mesh = jax.make_mesh((4,), ('model',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=False)
  linear = bx.Linear(
      graph.child('linear'),
      output_size=4,
      rng=rng,
      kernel_metadata={'sharding': ('model',)},
      bias_metadata={'sharding': ('model',)},
  )

  # Same init function for eval_shape and shard_map.
  def init_model(x):
    # No auto_fold_in_axes = all devices use same RNG sequence.
    params = rng.seed(bx.Params(), seed=42)
    _, params = linear(params, x)
    return params.finalized()

  x_sample = jnp.ones((1, 3))
  params_structure = jax.eval_shape(init_model, x_sample)
  param_specs = get_partition_spec(params_structure)

  # Init WITHOUT auto_fold_in_axes - all devices get same params.
  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=P(),
      out_specs=param_specs,
  )
  def init_sharded(x):
    return init_model(x)

  params = init_sharded(x_sample)

  # Apply to get output.
  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=(param_specs, P()),
      out_specs=(P('model', None), param_specs),
  )
  def apply_model(params, x):
    out, params = linear(params, x)
    return out, params

  out, _ = apply_model(params, x_sample)

  # === Verification ===
  # All devices should have identical weights (wrong for sharded init!).
  kernel_shards = params._data[
      ('root', 'linear', 'kernel')
  ].value.addressable_shards
  kernel_shard_data = [np.asarray(s.data) for s in kernel_shards]
  assert np.allclose(
      kernel_shard_data[0], kernel_shard_data[1]
  ), 'Without auto_fold_in_axes, all devices should have same weights'


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

  out = apply_ln(params, x)

  # === Verification ===
  global_mean = 2.25
  global_var = 0.3125
  expected = (x - global_mean) / jnp.sqrt(global_var + 1e-5)
  assert jnp.allclose(out, expected, atol=1e-5)


# =============================================================================
# Dropout with device-unique masks via auto_fold_in_axes
# =============================================================================


def test_dropout_with_auto_fold_in_axes_different_masks():
  """Dropout with auto_fold_in_axes produces different masks per device."""
  mesh = jax.make_mesh((4,), ('batch',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=P('batch', None),
      out_specs=P('batch', None),
  )
  def apply_dropout_with_auto_fold(x):
    # auto_fold_in_axes causes device-unique RNG.
    params = rng.seed(bx.Params(), seed=42)
    out, _ = dropout(params, x, is_training=True)
    return out

  x = jnp.ones((8, 16))
  out = apply_dropout_with_auto_fold(x)

  # === Verification ===
  out_per_device = out.reshape(4, 2, 16)
  zeros_per_device = [int(jnp.sum(out_per_device[i] == 0.0)) for i in range(4)]
  assert not all(
      z == zeros_per_device[0] for z in zeros_per_device
  ), 'Different devices should have different dropout masks with auto_fold_in_axes'


def test_dropout_without_auto_fold_in_axes_same_masks():
  """Dropout without auto_fold_in_axes produces same mask on all devices."""
  mesh = jax.make_mesh((4,), ('batch',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=False)
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=P('batch', None),
      out_specs=P('batch', None),
  )
  def apply_dropout_no_fold(x):
    # No auto_fold_in_axes = same RNG on all devices.
    params = rng.seed(bx.Params(), seed=42)
    out, _ = dropout(params, x, is_training=True)
    return out

  x = jnp.ones((8, 16))
  out = apply_dropout_no_fold(x)

  # === Verification ===
  out_per_device = out.reshape(4, 2, 16)
  for i in range(1, 4):
    assert jnp.allclose(
        out_per_device[0], out_per_device[i]
    ), 'Without auto_fold_in_axes, all devices should have same dropout mask'


# =============================================================================
# Mixed shard_map + vmap combinations
# =============================================================================


def test_shard_map_with_inner_vmap():
  """shard_map with named vmap inside produces unique keys."""
  mesh = jax.make_mesh((4,), ('x',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  @jax.shard_map(
      mesh=mesh,
      in_specs=P('x'),
      out_specs=P('x'),
  )
  def sharded_fn(y):
    # Named vmap inside shard_map.
    return jax.vmap(get_key, axis_name='v_inner')(y[0])[None]

  x = jnp.zeros((4, 2, 6))
  keys = sharded_fn(x)

  # All 8 keys should be unique (4 shards * 2 vmap positions).
  flat = keys.reshape(8, -1)
  for i in range(8):
    for j in range(i + 1, 8):
      assert not jnp.array_equal(flat[i], flat[j])


def test_jit_shard_map_with_inner_vmap():
  """JIT(shard_map(vmap)) produces unique keys."""
  mesh = jax.make_mesh((4,), ('x',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  @jax.jit
  @jax.shard_map(
      mesh=mesh,
      in_specs=P('x'),
      out_specs=P('x'),
  )
  def sharded_fn(y):
    return jax.vmap(get_key, axis_name='v_inner')(y[0])[None]

  x = jnp.zeros((4, 2, 6))
  keys = sharded_fn(x)

  # All 8 keys should be unique.
  flat = keys.reshape(8, -1)
  for i in range(8):
    for j in range(i + 1, 8):
      assert not jnp.array_equal(flat[i], flat[j])


def test_shard_map_jit_vmap():
  """shard_map(jit(vmap)) produces unique keys."""
  mesh = jax.make_mesh((4,), ('x',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  @jax.shard_map(
      mesh=mesh,
      in_specs=P('x'),
      out_specs=P('x'),
  )
  @jax.jit
  def sharded_fn(y):
    return jax.vmap(get_key, axis_name='v_inner')(y[0])[None]

  x = jnp.zeros((4, 2, 6))
  keys = sharded_fn(x)

  # All 8 keys should be unique.
  flat = keys.reshape(8, -1)
  for i in range(8):
    for j in range(i + 1, 8):
      assert not jnp.array_equal(flat[i], flat[j])


def test_scan_with_shard_map():
  """lax.scan with shard_map inside produces unique keys."""
  mesh = jax.make_mesh((2,), ('y',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  def scan_body(carry, chunk):
    i = carry
    sharded_result = jax.shard_map(
        lambda y: get_key(y)[None],
        mesh=mesh,
        in_specs=P('y'),
        out_specs=P('y'),
    )(chunk)
    return i + 1, sharded_result.flatten()

  x = jnp.zeros((4, 2, 6))
  _, keys = jax.lax.scan(scan_body, 0, x)

  # Keys should be different per shard within each step.
  for step in range(4):
    assert not jnp.array_equal(keys[step, 0], keys[step, 1])


def test_unnamed_vmap_inside_shard_map_fails():
  """Unnamed vmap inside shard_map fails with auto_fold_in_axes."""
  import pytest

  mesh = jax.make_mesh((4,), ('x',))

  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  @jax.shard_map(
      mesh=mesh,
      in_specs=P('x'),
      out_specs=P('x'),
  )
  def sharded_fn(y):
    # Unnamed vmap inside shard_map should fail.
    return jax.vmap(get_key)(y[0])[None]

  x = jnp.zeros((4, 2, 6))
  with pytest.raises(ValueError, match='Unnamed vmap detected'):
    sharded_fn(x)
