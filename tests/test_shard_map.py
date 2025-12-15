"""Tests for shard_map compatibility with sharded models."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
import blox as bx

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
  # Weight sharded across output dim (model parallelism).
  linear = bx.Linear(
    graph.child('linear'),
    output_size=16,
    kernel_metadata={'sharding': (None, 'model')},
    bias_metadata={'sharding': ('model',)},
  )

  # Initialize on host.
  x_sample = jnp.ones((1, 8))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))
  _, params = linear(params, x_sample)
  params = params.finalize()

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
  linear = bx.Linear(
    graph.child('linear'),
    output_size=4,
    kernel_metadata={'sharding': ()},
    bias_metadata={'sharding': ()},
  )

  # Initialize on host.
  x_sample = jnp.ones((1, 3))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))
  _, params = linear(params, x_sample)
  params = params.finalize()

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
  # Column parallel: shard output.
  layer1 = bx.Linear(
    graph.child('layer1'),
    output_size=16,
    kernel_metadata={'sharding': (None, 'model')},
    bias_metadata={'sharding': ('model',)},
  )
  # Row parallel: shard input.
  layer2 = bx.Linear(
    graph.child('layer2'),
    output_size=4,
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
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))
  _, params = mlp(params, x_sample)
  params = params.finalize()

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
# Sharded init with fold_in_axes produces device-unique params
# =============================================================================


def test_init_with_fold_in_axes_produces_different_params():
  """Init inside shard_map with fold_in_axes produces different params per device.

  When fold_in_axes is called, next_key() uses the folded key which incorporates
  the axis index, producing device-unique keys and thus different parameter
  values on each device.

  Key benefit: The same init function works for both eval_shape (to get specs)
  and actual sharded init. fold_in_axes/fold_out_axes are no-ops outside shard_map.
  """
  mesh = jax.make_mesh((4,), ('model',))

  graph = bx.Graph('root')
  linear = bx.Linear(
    graph.child('linear'),
    output_size=4,
    kernel_metadata={'sharding': ('model',)},
    bias_metadata={'sharding': ('model',)},
  )
  # Create Rng outside init_model so it can be reused.
  rng = bx.Rng(graph.child('rng'), seed=42)

  # Same init function works for eval_shape AND shard_map!
  def init_model(x):
    params = bx.Params(rng=rng).fold_in_axes('model')
    _, params = linear(params, x)
    # fold_out_axes to match pytree metadata with eval_shape.
    return params.fold_out_axes('model').finalize()

  x_sample = jnp.ones((1, 3))

  # Get structure - fold_in_axes/fold_out_axes are no-ops outside shard_map.
  params_structure = jax.eval_shape(init_model, x_sample)
  param_specs = get_partition_spec(params_structure)

  # Actual init inside shard_map - same function!
  @jax.jit
  @jax.shard_map(
    mesh=mesh,
    in_specs=P(),
    out_specs=param_specs,
  )
  def init_sharded(x):
    return init_model(x)

  params = init_sharded(x_sample)

  # Apply to get output from each device's params.
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
  # Each device should produce different output due to different weights.
  shards = out.addressable_shards
  shard_data = [np.asarray(s.data) for s in shards]
  assert not np.allclose(
    shard_data[0], shard_data[1]
  ), 'Different devices should have different params when using fold_in_axes'

  # Verify the weight values differ across shards.
  kernel_shards = params._data[
    ('root', 'linear', 'kernel')
  ].value.addressable_shards
  kernel_shard_data = [np.asarray(s.data) for s in kernel_shards]
  assert not np.allclose(
    kernel_shard_data[0], kernel_shard_data[1]
  ), 'Weights should differ across devices with fold_in_axes'


def test_init_without_fold_in_axes_produces_same_params():
  """Init inside shard_map WITHOUT fold_in_axes produces identical params.

  When fold_in_axes is not called, all devices use the same RNG sequence,
  resulting in identical parameters (which is usually wrong for sharded init).
  """
  mesh = jax.make_mesh((4,), ('model',))

  graph = bx.Graph('root')
  linear = bx.Linear(
    graph.child('linear'),
    output_size=4,
    kernel_metadata={'sharding': ('model',)},
    bias_metadata={'sharding': ('model',)},
  )
  # Create Rng outside init_model so it can be reused.
  rng = bx.Rng(graph.child('rng'), seed=42)

  # Same init function for eval_shape and shard_map.
  def init_model(x):
    # No fold_in_axes = all devices use same RNG sequence.
    params = bx.Params(rng=rng)
    _, params = linear(params, x)
    return params.finalize()

  x_sample = jnp.ones((1, 3))
  params_structure = jax.eval_shape(init_model, x_sample)
  param_specs = get_partition_spec(params_structure)

  # Init WITHOUT fold_in_axes - all devices get same params.
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
  ), 'Without fold_in_axes, all devices should have same weights'


# =============================================================================
# LayerNorm with cross-device aggregation
# =============================================================================


def test_layernorm_cross_device():
  """Test LayerNorm with axis_name aggregates mean/var across devices."""
  mesh = jax.make_mesh((4,), ('batch',))

  graph = bx.Graph('root')
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

  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

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
# Dropout with device-unique masks via fold_in_axes
# =============================================================================


def test_dropout_with_fold_in_axes_different_masks():
  """Dropout with fold_in_axes produces different masks per device."""
  mesh = jax.make_mesh((4,), ('batch',))

  graph = bx.Graph('root')
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5)

  @jax.jit
  @jax.shard_map(
    mesh=mesh,
    in_specs=P('batch', None),
    out_specs=P('batch', None),
  )
  def apply_dropout_with_fold_in(x):
    # fold_in_axes causes device-unique RNG.
    params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42)).fold_in_axes(
      'batch'
    )
    out, _ = dropout(params, x, is_training=True)
    return out

  x = jnp.ones((8, 16))
  out = apply_dropout_with_fold_in(x)

  # === Verification ===
  out_per_device = out.reshape(4, 2, 16)
  zeros_per_device = [int(jnp.sum(out_per_device[i] == 0.0)) for i in range(4)]
  assert not all(
    z == zeros_per_device[0] for z in zeros_per_device
  ), 'Different devices should have different dropout masks with fold_in_axes'


def test_dropout_without_fold_in_axes_same_masks():
  """Dropout without fold_in_axes produces same mask on all devices."""
  mesh = jax.make_mesh((4,), ('batch',))

  graph = bx.Graph('root')
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5)

  @jax.jit
  @jax.shard_map(
    mesh=mesh,
    in_specs=P('batch', None),
    out_specs=P('batch', None),
  )
  def apply_dropout_no_fold(x):
    # No fold_in_axes = same RNG on all devices.
    params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))
    out, _ = dropout(params, x, is_training=True)
    return out

  x = jnp.ones((8, 16))
  out = apply_dropout_no_fold(x)

  # === Verification ===
  out_per_device = out.reshape(4, 2, 16)
  for i in range(1, 4):
    assert jnp.allclose(
      out_per_device[0], out_per_device[i]
    ), 'Without fold_in_axes, all devices should have same dropout mask'
