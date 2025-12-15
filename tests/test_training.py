import jax
import jax.numpy as jnp
import blox as bx
import chex


def test_rng_updates_during_training():
  """Verifies that non-trainable state (RNG) is correctly updated after grad."""

  # Setup the model graph and layer.
  graph = bx.Graph('root')
  model = bx.Linear(graph.child('linear'), output_size=1)

  # Initialize data and parameters.
  x = jnp.ones((1, 5))
  y = jnp.ones((1, 1))

  # Create params with an Rng.
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))

  # RNG counter path is under the Rng module's graph path.
  counter_path = ('root', 'rng', 'counter')

  # Counter is not created until first next_key() call.
  assert counter_path not in params._data

  # Run the initialization pass to create weights (and RNG state).
  _, params = model(params, x)

  # NOW we finalize, to prevent accidental creation during training.
  params = params.finalize()

  initial_counter = params._data[counter_path].value

  # Counter increments once for kernel (lecun_normal needs key).
  # Bias uses zeros() which works with key=None, so no increment.
  assert initial_counter == 1

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
    def __call__(self, p, x):
      # Manually consume a key to simulate dropout.
      _, new_p = p.next_key()
      return x, new_p

  dropout = MockDropout(graph.child('drop'))

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
  layer1 = bx.Linear(graph.child('layer1'), output_size=32)
  layer2 = bx.Linear(graph.child('layer2'), output_size=32)
  layer3 = bx.Linear(graph.child('layer3'), output_size=8)

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
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))
  _, params = forward(params, x)
  params = params.finalize()

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
  linear1 = bx.Linear(graph.child('linear1'), output_size=16)
  linear2 = bx.Linear(graph.child('linear2'), output_size=8)
  dropout = bx.Dropout(graph.child('dropout'), rate=0.5)

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
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))
  _, params = forward(params, x)
  params = params.finalize()

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
