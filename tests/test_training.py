import blox as bx
import chex
import jax
import jax.numpy as jnp


def test_rng_updates_during_training():
  """Verifies that non-trainable state (RNG) is correctly updated after grad."""
  # Setup the model graph and layer.
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  model = bx.Linear(graph.child('linear'), output_size=1, rng=rng)

  # Initialize data and parameters.
  x = jnp.ones((1, 5))
  y = jnp.ones((1, 1))

  # Create params with an Rng.
  params = rng.seed(bx.Params(), seed=42)

  # RNG counter path is under the Rng module's graph path.
  counter_path = ('root', 'rng', 'counter')

  # Counter is created at seed initialization, starting at 0.
  assert counter_path in params._data
  assert params._data[counter_path].value == 0

  # Run the initialization pass to create weights (and RNG state).
  _, params = model(params, x)

  # NOW we finalize, to prevent accidental creation during training.
  params = params.locked()

  initial_counter = params._data[counter_path].value

  # We always generate a key for every param to ensure consistent
  # initialization order regardless of which params use random init.
  assert initial_counter == 2

  @jax.jit
  def train_step(p, inputs, targets):
    trainable, non_trainable = p.split()

    def loss_fn(t, nt):
      full_p = t.merge(nt)

      # Run the forward pass which might increment the RNG.
      pred, new_p = model(full_p, inputs)

      # Extract the updated non-trainable state.
      _, new_nt = new_p.split()

      return jnp.mean((pred - targets) ** 2), new_nt

    # Use grad with has_aux to get gradients and the updated state.
    grads, new_nt = jax.grad(loss_fn, has_aux=True)(trainable, non_trainable)

    # Apply simple SGD updates.
    new_t = jax.tree.map(lambda w, g: w - 0.01 * g, trainable, grads)

    # Merge the updated weights with the updated non-trainable state.
    return new_t.merge(new_nt)

  # Run the training step.
  new_params = train_step(params, x, y)

  # Ensure the counter state is preserved or updated correctly.
  assert new_params._data[counter_path].value == initial_counter

  # Define a mock layer that consumes RNG during the forward pass.
  class MockDropout(bx.Module):

    def __init__(self, g, rng):
      super().__init__(g)
      self.rng = rng

    def __call__(self, p, x):
      # Manually consume a key to simulate dropout.
      _, new_p = self.rng(p)
      return x, new_p

  dropout = MockDropout(graph.child('drop'), rng)

  @jax.jit
  def dropout_train_step(p, inputs):
    t, nt = p.split()

    def loss(t_inner, nt_inner):
      full = t_inner.merge(nt_inner)
      # This call increments the internal RNG counter.
      _, new_full = dropout(full, inputs)
      _, new_nt = new_full.split()
      return 0.0, new_nt

    _, new_nt_out = jax.grad(loss, has_aux=True)(t, nt)
    return t.merge(new_nt_out)

  # Run the dropout step.
  params_after_dropout = dropout_train_step(params, x)

  # Verify that the RNG counter has incremented.
  assert params_after_dropout._data[counter_path].value == initial_counter + 1


def test_checkpoint_produces_correct_gradients():
  """Verifies that jax.checkpoint works correctly with blox modules.

  jax.checkpoint (remat) trades compute for memory by recomputing activations
  during the backward pass. The typical pattern is to checkpoint individual
  layers/blocks within a larger network - this saves the activations *between*
  blocks but recomputes activations *within* checkpointed blocks.

  Here we checkpoint layer2 in a 3-layer network:
  - layer1 output: saved (needed as input to checkpointed block)
  - layer2 intermediates: recomputed during backward pass
  - layer3 output: saved
  """
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  layer1 = bx.Linear(graph.child('layer1'), output_size=32, rng=rng)
  layer2 = bx.Linear(graph.child('layer2'), output_size=32, rng=rng)
  layer3 = bx.Linear(graph.child('layer3'), output_size=8, rng=rng)

  def forward(p, inputs):
    h, p = layer1(p, inputs)
    h = jax.nn.relu(h)
    h, p = layer2(p, h)
    h = jax.nn.relu(h)
    out, p = layer3(p, h)
    return out, p

  # Checkpointed version: only layer2 block is checkpointed.
  @jax.checkpoint
  def layer2_block(p, h):
    h, p = layer2(p, h)
    return jax.nn.relu(h), p

  def forward_checkpointed(p, inputs):
    h, p = layer1(p, inputs)
    h = jax.nn.relu(h)
    # This block's intermediates will be recomputed during backprop.
    h, p = layer2_block(p, h)
    out, p = layer3(p, h)
    return out, p

  x = jnp.ones((4, 8))
  y = jnp.ones((4, 8))

  # Initialize params by running forward.
  params = rng.seed(bx.Params(), seed=42)
  _, params = forward(params, x)
  params = params.locked()

  trainable, non_trainable = params.split()

  # Compute gradients without checkpoint.
  def loss_fn(t, nt, inputs, targets):
    p = t.merge(nt)
    pred, _ = forward(p, inputs)
    return jnp.mean((pred - targets) ** 2)

  grads_normal = jax.grad(loss_fn)(trainable, non_trainable, x, y)

  # Compute gradients with checkpoint.
  def loss_fn_checkpointed(t, nt, inputs, targets):
    p = t.merge(nt)
    pred, _ = forward_checkpointed(p, inputs)
    return jnp.mean((pred - targets) ** 2)

  grads_checkpointed = jax.grad(loss_fn_checkpointed)(
      trainable, non_trainable, x, y
  )

  # Gradients should match exactly.
  chex.assert_trees_all_close(grads_normal, grads_checkpointed)


def test_checkpoint_with_dropout():
  """Verifies checkpoint works with RNG-consuming layers like dropout.

  When checkpoint recomputes the forward pass during backprop, the RNG must
  produce the same random values as the original forward pass. blox's
  counter-based RNG ensures this reproducibility - the counter value
  deterministically selects which random values are generated.

  Here we checkpoint a block containing dropout. The key insight is that
  the RNG counter is part of the Params pytree, so it gets "saved" at the
  checkpoint boundary and "restored" during recomputation.
  """
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  linear1 = bx.Linear(graph.child('linear1'), output_size=16, rng=rng)
  linear2 = bx.Linear(graph.child('linear2'), output_size=8, rng=rng)
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

  def forward(p, inputs):
    h, p = linear1(p, inputs)
    h = jax.nn.relu(h)
    h, p = dropout(p, h, is_training=True)
    out, p = linear2(p, h)
    return out, p

  # Checkpoint the dropout block.
  @jax.checkpoint
  def dropout_block(p, h):
    h, p = dropout(p, h, is_training=True)
    return h, p

  def forward_checkpointed(p, inputs):
    h, p = linear1(p, inputs)
    h = jax.nn.relu(h)
    h, p = dropout_block(p, h)
    out, p = linear2(p, h)
    return out, p

  x = jnp.ones((4, 8))
  y = jnp.ones((4, 8))

  # Initialize params by running forward.
  params = rng.seed(bx.Params(), seed=42)
  _, params = forward(params, x)
  params = params.locked()

  trainable, non_trainable = params.split()

  # Compute gradients without checkpoint.
  def loss_fn(t, nt, inputs, targets):
    p = t.merge(nt)
    pred, new_p = forward(p, inputs)
    _, new_nt = new_p.split()
    return jnp.mean((pred - targets) ** 2), new_nt

  grads_normal, nt_normal = jax.grad(loss_fn, has_aux=True)(
      trainable, non_trainable, x, y
  )

  # Compute gradients with checkpoint.
  def loss_fn_checkpointed(t, nt, inputs, targets):
    p = t.merge(nt)
    pred, new_p = forward_checkpointed(p, inputs)
    _, new_nt = new_p.split()
    return jnp.mean((pred - targets) ** 2), new_nt

  grads_checkpointed, nt_checkpointed = jax.grad(
      loss_fn_checkpointed, has_aux=True
  )(trainable, non_trainable, x, y)

  # Gradients should match.
  chex.assert_trees_all_close(grads_normal, grads_checkpointed)

  # RNG counter should be updated the same way.
  counter_path = ('root', 'rng', 'counter')
  assert (
      nt_normal._data[counter_path].value
      == nt_checkpointed._data[counter_path].value
  )


def test_rng_get_set_seed():
  """Verifies get_seed and seed methods on Rng."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  params = rng.seed(bx.Params(), seed=42)

  # get_seed returns the seed key.
  key1 = rng.get_seed(params)
  assert key1 is not None

  # get_seed returns the same key on subsequent calls.
  key2 = rng.get_seed(params)
  assert jnp.array_equal(key1, key2)

  # seed() with seed= returns params with updated key.
  new_key = jax.random.key(999)
  new_params = rng.seed(params, seed=new_key)

  key3 = rng.get_seed(new_params)
  assert jnp.array_equal(key3, new_key)
  assert not jnp.array_equal(key3, key1)

  # Original params unchanged.
  key_orig = rng.get_seed(params)
  assert jnp.array_equal(key_orig, key1)


def test_rng_get_set_counter():
  """Verifies get_counter and seed(counter=) methods on Rng."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  params = rng.seed(bx.Params(), seed=42)

  # get_counter returns counter value (starts at 0).
  counter1 = rng.get_counter(params)
  assert counter1 == 0

  # Calling rng increments the counter.
  _, params = rng(params)
  counter2 = rng.get_counter(params)
  assert counter2 == 1

  # seed(counter=) returns params with updated counter.
  new_params = rng.seed(params, counter=100)
  counter3 = rng.get_counter(new_params)
  assert counter3 == 100

  # Original params unchanged.
  counter_orig = rng.get_counter(params)
  assert counter_orig == 1


def test_rng_reseed_pattern():
  """Verifies the reseed pattern using seed()."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  params = rng.seed(bx.Params(), seed=42)

  # Initialize and consume some keys.
  _, params = rng(params)
  _, params = rng(params)
  _, params = rng(params)
  params = params.locked()

  counter_before = rng.get_counter(params)
  assert counter_before == 3

  # Reseed: set new key and reset counter in one call.
  new_key = jax.random.key(999)
  reseeded = rng.seed(params, seed=new_key, counter=0)

  counter_after = rng.get_counter(reseeded)
  assert counter_after == 0

  key_after = rng.get_seed(reseeded)
  assert jnp.array_equal(key_after, new_key)


def test_auto_fold_in_axes_produces_different_keys():
  """Verifies auto_fold_in_axes produces different RNG keys per batch element."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  keys = jax.vmap(get_key, axis_name='batch')(jnp.ones((4, 1)))

  for i in range(1, 4):
    assert not jnp.array_equal(keys[0], keys[i])


def test_auto_fold_in_axes_disabled_produces_same_keys():
  """Verifies auto_fold_in_axes=False produces same RNG keys across batch."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=False)

  def get_key(x):
    params = rng.seed(bx.Params(), seed=42)
    key, _ = rng(params)
    return key

  keys = jax.vmap(get_key, axis_name='batch')(jnp.ones((4, 1)))

  for i in range(1, 4):
    assert jnp.array_equal(keys[0], keys[i])


def test_auto_fold_in_axes_noop_outside_vmap():
  """Verifies auto_fold_in_axes is a no-op outside vmap."""
  graph = bx.Graph('root')
  rng1 = bx.Rng(graph.child('rng1'), auto_fold_in_axes=True)
  rng2 = bx.Rng(graph.child('rng2'), auto_fold_in_axes=False)

  params = rng1.seed(bx.Params(), seed=42)
  params = rng2.seed(params, seed=42)

  # Outside vmap, both should produce the same keys.
  key1, _ = rng1(params)
  key2, _ = rng2(params)

  assert jnp.array_equal(key1, key2)


def test_module_param_path():
  """Verifies Module.param_path() returns correct paths."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  linear = bx.Linear(
      graph.child('encoder').child('layer'), output_size=4, rng=rng
  )

  # param_path returns the full tuple path for a parameter name.
  kernel_path = linear.param_path('kernel')
  bias_path = linear.param_path('bias')

  assert kernel_path == ('root', 'encoder', 'layer', 'kernel')
  assert bias_path == ('root', 'encoder', 'layer', 'bias')


def test_params_len_and_contains():
  """Verifies Params.__len__ and __contains__ work correctly."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  linear = bx.Linear(graph.child('linear'), output_size=4, rng=rng)

  x = jnp.ones((2, 3))
  params = rng.seed(bx.Params(), seed=42)
  _, params = linear(params, x)
  params = params.locked()

  # __len__ returns number of parameters.
  assert len(params) == 4  # kernel, bias, seed, counter

  # __contains__ checks for path existence.
  assert ('root', 'linear', 'kernel') in params
  assert ('root', 'linear', 'bias') in params
  assert ('root', 'rng', 'seed') in params
  assert ('root', 'rng', 'counter') in params
  assert ('root', 'nonexistent') not in params
