import blox as bx
import jax
import jax.numpy as jnp


def test_linear_shapes():
  """Verifies shape inference and parameter creation."""
  graph = bx.Graph('root')
  # Linear layer with 10 outputs.
  layer = bx.Linear(graph.child('linear'), output_size=10)

  # Input has 5 features.
  x = jnp.ones((2, 5))
  # Initialize params with a seed.
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, params = layer(params, x)

  # Check output.
  assert y.shape == (2, 10)

  # Check params existed.
  frozen = params.finalized()
  # Path is ('root', 'linear', 'kernel') because graph was "root" -> child("linear").
  # Note: Access .value because _data stores Param objects.
  kernel_shape = frozen._data[('root', 'linear', 'kernel')].value.shape
  bias_shape = frozen._data[('root', 'linear', 'bias')].value.shape

  assert kernel_shape == (5, 10)
  assert bias_shape == (10,)


def test_linear_learning():
  """Verifies that gradients propagate through the layer."""
  graph = bx.Graph('net')
  layer = bx.Linear(graph.child('linear'), output_size=1, use_bias=False)

  x = jnp.array([[1.0, 2.0]])
  y_target = jnp.array([[5.0]])

  # Initialize params with a seed.
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))

  # Initialize.
  _, params = layer(params, x)
  frozen_params = params.finalized()

  # Train step.
  @jax.jit
  def step(p):
    # Split params: We only want gradients for trainable weights.
    # The RNG state (and any non-trainable state) goes into 'non_trainable'.
    trainable, non_trainable = p.split()

    def loss(t):
      # Merge back to run the model (model needs full state)
      full_params = t.merge(non_trainable)
      pred, _ = layer(full_params, x)
      return jnp.mean((pred - y_target) ** 2)

    # Grad w.r.t 'trainable' only
    grads = jax.grad(loss)(trainable)

    # Update
    new_trainable = jax.tree.map(lambda w, g: w - 0.1 * g, trainable, grads)

    # Return full state (merged)
    return new_trainable.merge(non_trainable)

  # Train for a few steps.
  curr = frozen_params
  for _ in range(20):
    curr = step(curr)

  pred, _ = layer(curr, x)
  assert jnp.allclose(pred, y_target, atol=1e-2)


def test_root_node_protection():
  """Verifies that modules cannot be bound directly to the root graph node."""
  graph = bx.Graph('root')

  # Attempting to bind to root should fail.
  try:
    bx.Linear(graph, output_size=10)
    # If the line above doesn't raise, we force a failure.
    raise AssertionError('Module allowed binding to root graph node.')
  except ValueError as e:
    # Verify we caught the correct error message.
    assert 'root graph node' in str(e)

  # Binding to a child should succeed.
  try:
    bx.Linear(graph.child('safe_layer'), output_size=10)
  except ValueError:
    raise AssertionError('Module failed to bind to a valid child node.')


def test_sequential_chaining():
  """Verifies Sequential correctly chains layers and functions."""
  graph = bx.Graph('root')
  model = bx.Sequential(
      graph.child('seq'),
      [
          bx.Linear(graph.child('l1'), output_size=10),
          jax.nn.relu,
          bx.Linear(graph.child('l2'), output_size=5),
      ],
  )

  x = jnp.ones((2, 20))  # Batch=2, Features=20
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, params = model(params, x)

  assert y.shape == (2, 5)


def test_sequential_empty():
  """Verifies Sequential with empty layers acts as identity."""
  graph = bx.Graph('root')
  model = bx.Sequential(graph.child('seq'), [])

  x = jnp.ones((2, 5))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, _ = model(params, x)
  assert jnp.allclose(y, x)


def test_sequential_nested():
  """Verifies nested Sequential modules."""
  graph = bx.Graph('root')
  inner = bx.Sequential(
      graph.child('inner'), [bx.Linear(graph.child('l1'), output_size=5)]
  )
  outer = bx.Sequential(graph.child('outer'), [inner, jax.nn.relu])

  x = jnp.ones((2, 10))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, params = outer(params, x)
  assert y.shape == (2, 5)


def test_sequential_lambda():
  """Verifies Sequential with a lambda layer."""
  graph = bx.Graph('root')
  model = bx.Sequential(graph.child('seq'), [lambda x: x * 2])

  x = jnp.ones((2, 5))
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  y, _ = model(params, x)
  assert jnp.allclose(y, x * 2)
