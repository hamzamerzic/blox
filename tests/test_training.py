import jax
import jax.numpy as jnp
import blox as bx


def test_rng_updates_during_training():
  """Verifies that non-trainable state (RNG) is correctly updated after grad."""

  # Setup the model graph and layer.
  graph = bx.Graph('root')
  model = bx.Linear(graph.child('linear'), output_size=1)

  # Initialize data and parameters.
  x = jnp.ones((1, 5))
  y = jnp.ones((1, 1))

  # FIX: Do not finalize yet! We need to create parameters first.
  params = bx.Params(seed=42)

  # Ensure the RNG counter starts at 0.
  assert params._data['rng'].value[1] == 0

  # Run the initialization pass to create weights.
  _, params = model(params, x)

  # NOW we finalize, to prevent accidental creation during training.
  params = params.finalize()

  initial_counter = params._data['rng'].value[1]

  # The counter increments twice during init (once for weights, once for bias).
  assert initial_counter == 2

  @jax.jit
  def train_step(p, inputs, targets):
    trainable, non_trainable = p.split_trainable()

    def loss_fn(t, nt):
      full_p = t.merge(nt)

      # Run the forward pass which might increment the RNG.
      pred, new_p = model(full_p, inputs)

      # Extract the updated non-trainable state.
      _, new_nt = new_p.split_trainable()

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
  assert new_params._data['rng'].value[1] == initial_counter

  # Define a mock layer that consumes RNG during the forward pass.
  class MockDropout(bx.Module):
    def __call__(self, p, x):
      # Manually consume a key to simulate dropout.
      _, new_p = p.next_key()
      return x, new_p

  dropout = MockDropout(bx.Graph('drop'))

  @jax.jit
  def dropout_train_step(p, inputs):
    t, nt = p.split_trainable()

    def loss(t_inner, nt_inner):
      full = t_inner.merge(nt_inner)
      # This call increments the internal RNG counter.
      _, new_full = dropout(full, inputs)
      _, new_nt = new_full.split_trainable()
      return 0.0, new_nt

    _, new_nt_out = jax.grad(loss, has_aux=True)(t, nt)
    return t.merge(new_nt_out)

  # Run the dropout step.
  params_after_dropout = dropout_train_step(params, x)

  # Verify that the RNG counter has incremented from 1 to 2.
  assert params_after_dropout._data['rng'].value[1] == initial_counter + 1
