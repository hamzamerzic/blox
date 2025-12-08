import jax
import jax.numpy as jnp
import blox as bx


def test_linear_shapes():
  """Verifies shape inference and parameter creation."""
  graph = bx.Graph('root')
  # Linear layer with 10 outputs.
  layer = bx.Linear(graph.child('linear'), output_size=10)

  # Input has 5 features.
  x = jnp.ones((2, 5))
  # Initialize params with a seed.
  params = bx.Params(seed=0)

  y, params = layer(params, x)

  # Check output.
  assert y.shape == (2, 10)

  # Check params existed.
  frozen = params.finalize()
  # Path is 'root/linear/w' because graph was "root" -> child("linear").
  # Note: Access .value because _data stores Variable objects.
  w_shape = frozen._data['root/linear/w'].value.shape
  b_shape = frozen._data['root/linear/b'].value.shape

  assert w_shape == (5, 10)
  assert b_shape == (10,)


def test_linear_learning():
  """Verifies that gradients propagate through the layer."""
  graph = bx.Graph('net')
  layer = bx.Linear(graph.child('linear'), output_size=1, with_bias=False)

  x = jnp.array([[1.0, 2.0]])
  y_target = jnp.array([[5.0]])

  # Initialize params with a seed.
  params = bx.Params(seed=42)

  # Initialize.
  _, params = layer(params, x)
  frozen_params = params.finalize()

  # Train step.
  @jax.jit
  def step(p):
    # 1. Split params: We only want gradients for trainable weights.
    #    The RNG state (and any frozen config) goes into 'non_trainable'.
    trainable, non_trainable = p.partition(lambda _, v: v.trainable)

    def loss(t):
      # 2. Merge back to run the model (model needs full state)
      full_params = t.merge(non_trainable)
      pred, _ = layer(full_params, x)
      return jnp.mean((pred - y_target) ** 2)

    # 3. Grad w.r.t 'trainable' only
    grads = jax.grad(loss)(trainable)

    # 4. Update
    new_trainable = jax.tree.map(lambda w, g: w - 0.1 * g, trainable, grads)

    # 5. Return full state (merged)
    return new_trainable.merge(non_trainable)

  # Train for a few steps.
  curr = frozen_params
  for _ in range(20):
    curr = step(curr)

  pred, _ = layer(curr, x)
  assert jnp.allclose(pred, y_target, atol=1e-2)
