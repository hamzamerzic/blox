import blox as bx
import jax
import jax.numpy as jnp


def test_linear_shapes():
  """Verifies shape inference and parameter creation."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  # Linear layer with 10 outputs.
  layer = bx.Linear(graph.child('linear'), output_size=10, rng=rng)

  # Input has 5 features.
  x = jnp.ones((2, 5))
  # Initialize params with a seed.
  params = rng.seed(bx.Params(), seed=0)

  y, params = layer(params, x)

  # Check output.
  assert y.shape == (2, 10)

  # Check params existed.
  frozen = params.locked()
  # Path is ('root', 'linear', 'kernel') because graph was "root" -> child("linear").
  # Note: Access .value because _data stores Param objects.
  kernel_shape = frozen._data[('root', 'linear', 'kernel')].value.shape
  bias_shape = frozen._data[('root', 'linear', 'bias')].value.shape

  assert kernel_shape == (5, 10)
  assert bias_shape == (10,)


def test_linear_learning():
  """Verifies that gradients propagate through the layer."""
  graph = bx.Graph('net')
  rng = bx.Rng(graph.child('rng'))
  layer = bx.Linear(
      graph.child('linear'), output_size=1, rng=rng, use_bias=False
  )

  x = jnp.array([[1.0, 2.0]])
  y_target = jnp.array([[5.0]])

  # Initialize params with a seed.
  params = rng.seed(bx.Params(), seed=42)

  # Initialize.
  _, params = layer(params, x)
  frozen_params = params.locked()

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
  rng = bx.Rng(graph.child('rng'))

  # Attempting to bind to root should fail.
  try:
    bx.Linear(graph, output_size=10, rng=rng)
    # If the line above doesn't raise, we force a failure.
    raise AssertionError('Module allowed binding to root graph node.')
  except ValueError as e:
    # Verify we caught the correct error message.
    assert 'root graph node' in str(e)

  # Binding to a child should succeed.
  try:
    bx.Linear(graph.child('safe_layer'), output_size=10, rng=rng)
  except ValueError:
    raise AssertionError('Module failed to bind to a valid child node.')


def test_sequential_chaining():
  """Verifies Sequential correctly chains layers and functions."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  model = bx.Sequential(
      graph.child('seq'),
      [
          bx.Linear(graph.child('l1'), output_size=10, rng=rng),
          jax.nn.relu,
          bx.Linear(graph.child('l2'), output_size=5, rng=rng),
      ],
  )

  x = jnp.ones((2, 20))  # Batch=2, Features=20
  params = rng.seed(bx.Params(), seed=0)

  y, params = model(params, x)

  assert y.shape == (2, 5)


def test_sequential_empty():
  """Verifies Sequential with empty layers acts as identity."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  model = bx.Sequential(graph.child('seq'), layers=[])

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)

  y, _ = model(params, x)
  assert jnp.allclose(y, x)


def test_sequential_nested():
  """Verifies nested Sequential modules."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  inner = bx.Sequential(
      graph.child('inner'),
      [bx.Linear(graph.child('l1'), output_size=5, rng=rng)],
  )
  outer = bx.Sequential(graph.child('outer'), [inner, jax.nn.relu])

  x = jnp.ones((2, 10))
  params = rng.seed(bx.Params(), seed=0)

  y, params = outer(params, x)
  assert y.shape == (2, 5)


def test_sequential_lambda():
  """Verifies Sequential with a lambda layer."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  model = bx.Sequential(graph.child('seq'), [lambda x: x * 2])

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)

  y, _ = model(params, x)
  assert jnp.allclose(y, x * 2)


# =============================================================================
# set_param Tests
# =============================================================================


def test_set_param_value():
  """Verifies set_param can update parameter values."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  layer = bx.Linear(graph.child('linear'), output_size=3, rng=rng)

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)
  _, params = layer(params, x)
  params = params.locked()

  # Update kernel with new values.
  new_kernel = jnp.ones((5, 3))
  params = layer.set_param(params, 'kernel', new_kernel)

  assert jnp.allclose(
      params._data[('root', 'linear', 'kernel')].value, new_kernel
  )


def test_set_param_trainable():
  """Verifies set_param can update trainable flag."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  layer = bx.Linear(graph.child('linear'), output_size=3, rng=rng)

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)
  _, params = layer(params, x)
  params = params.locked()

  # Initially trainable.
  assert params._data[('root', 'linear', 'kernel')].trainable is True

  # Freeze the parameter.
  params = layer.set_param(params, 'kernel', None, trainable=False)

  assert params._data[('root', 'linear', 'kernel')].trainable is False
  # Value should be unchanged.
  assert params._data[('root', 'linear', 'kernel')].value is not None


def test_set_param_metadata():
  """Verifies set_param can update metadata."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  layer = bx.Linear(graph.child('linear'), output_size=3, rng=rng)

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)
  _, params = layer(params, x)
  params = params.locked()

  # Add metadata.
  params = layer.set_param(params, 'kernel', None, metadata={'tag': 'lora'})

  assert params._data[('root', 'linear', 'kernel')].metadata['tag'] == 'lora'


def test_set_param_metadata_merges():
  """Verifies set_param merges metadata with existing."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  layer = bx.Linear(
      graph.child('linear'),
      output_size=3,
      rng=rng,
      kernel_metadata={'sharding': (None, 'model')},
  )

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)
  _, params = layer(params, x)
  params = params.locked()

  # Add more metadata - should merge.
  params = layer.set_param(params, 'kernel', None, metadata={'tag': 'lora'})

  meta = params._data[('root', 'linear', 'kernel')].metadata
  assert meta['sharding'] == (None, 'model')  # Preserved.
  assert meta['tag'] == 'lora'  # Added.


def test_set_param_requires_something():
  """Verifies set_param fails if nothing is provided."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  layer = bx.Linear(graph.child('linear'), output_size=3, rng=rng)

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)
  _, params = layer(params, x)
  params = params.locked()

  try:
    layer.set_param(params, 'kernel', None)
    assert False, 'Should have raised ValueError'
  except ValueError as e:
    assert 'At least one' in str(e)


# =============================================================================
# Params.__setitem__ and unlocked() Tests
# =============================================================================


def test_params_setitem():
  """Verifies __setitem__ can set a Param directly."""
  params = bx.Params()

  kernel = jnp.ones((5, 3))
  params[('net', 'linear', 'kernel')] = bx.Param(kernel, trainable=True)

  assert ('net', 'linear', 'kernel') in params
  assert jnp.allclose(params[('net', 'linear', 'kernel')].value, kernel)


def test_params_setitem_requires_param():
  """Verifies __setitem__ requires a Param object."""
  params = bx.Params()

  try:
    params[('net', 'linear', 'kernel')] = jnp.ones((5, 3))  # type: ignore[arg-type]
    assert False, 'Should have raised TypeError'
  except TypeError as e:
    assert 'Expected Param' in str(e)


def test_params_setitem_fails_when_locked():
  """Verifies __setitem__ fails on locked params."""
  params = bx.Params().locked()

  try:
    params[('net', 'linear', 'kernel')] = bx.Param(jnp.ones((5, 3)))
    assert False, 'Should have raised RuntimeError'
  except RuntimeError as e:
    assert 'locked' in str(e).lower()


def test_params_unlocked():
  """Verifies unlocked() allows new params to be added."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  layer = bx.Linear(graph.child('linear'), output_size=3, rng=rng)

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)
  _, params = layer(params, x)
  params = params.locked()

  # This would fail on locked params.
  try:
    params[('net', 'new', 'param')] = bx.Param(jnp.ones((2,)))
    assert False, 'Should have raised RuntimeError'
  except RuntimeError:
    pass

  # But works after unlocked().
  params = params.unlocked()
  params[('net', 'new', 'param')] = bx.Param(jnp.ones((2,)))

  assert ('net', 'new', 'param') in params


def test_params_unlocked_allows_get_param():
  """Verifies unlocked() allows new params via get_param."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  layer1 = bx.Linear(graph.child('linear1'), output_size=3, rng=rng)

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)
  _, params = layer1(params, x)
  params = params.locked()

  # Create a new layer after locking.
  layer2 = bx.Linear(graph.child('linear2'), output_size=2, rng=rng)

  # Would fail on locked.
  try:
    _, params = layer2(params, x)
    assert False, 'Should have raised KeyError'
  except KeyError:
    pass

  # Works after unlocked.
  params = params.unlocked()
  _, params = layer2(params, x)

  assert ('root', 'linear2', 'kernel') in params


# =============================================================================
# get_param without shape/init Tests
# =============================================================================


def test_get_param_existing_without_shape_init():
  """Verifies get_param can retrieve existing params without shape/init."""
  graph = bx.Graph('root')
  rng = bx.Rng(graph.child('rng'))
  layer = bx.Linear(graph.child('linear'), output_size=3, rng=rng)

  x = jnp.ones((2, 5))
  params = rng.seed(bx.Params(), seed=0)
  _, params = layer(params, x)
  params = params.locked()

  # Get existing kernel without specifying shape/init.
  kernel, _ = layer.get_param(params, 'kernel')

  assert kernel.shape == (5, 3)


def test_get_param_missing_without_shape_init():
  """Verifies get_param raises KeyError for missing params without shape/init."""
  graph = bx.Graph('root')
  layer = bx.Linear(graph.child('linear'), output_size=3, rng=None)

  params = bx.Params()

  try:
    layer.get_param(params, 'kernel')
    assert False, 'Should have raised KeyError'
  except KeyError as e:
    assert 'not found' in str(e)
    assert 'shape and init' in str(e)
