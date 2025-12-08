import jax.numpy as jnp
import blox as bx
import pytest


def test_lstm_sequence_shapes():
  """Verifies input [B, T, D] -> output [B, T, Hidden]."""
  graph = bx.Graph('root')
  lstm = bx.LSTM(graph.child('lstm'), hidden_size=10)

  # Batch=2, Time=5, Dim=4
  inputs = jnp.ones((2, 5, 4))
  params = bx.Params(seed=0)

  # Explicitly initialize state.
  state, params = lstm.initial_state(params, inputs)

  # Call returns ((out, state), params).
  ((out, state), params) = lstm(params, inputs, state)

  # Output should match batch and time dims.
  assert out.shape == (2, 5, 10)
  # Final state is [B, Hidden].
  assert state.hidden.shape == (2, 10)


def test_lstm_step_signature():
  """Verifies step returns strict tuple structure: ((out, state), params)."""
  graph = bx.Graph('root')
  lstm = bx.LSTM(graph.child('lstm'), hidden_size=10)

  inputs = jnp.ones((2, 4))
  params = bx.Params(seed=0)

  # Initialize state manually.
  state, params = lstm.initial_state(params, inputs)

  # Step call.
  ret = lstm.step(params, inputs, state)

  # Check structure: ((out, state), params).
  assert isinstance(ret, tuple)
  assert len(ret) == 2
  ((out, new_state), new_params) = ret

  assert out.shape == (2, 10)
  assert isinstance(new_state, bx.LSTMState)
  assert isinstance(new_params, bx.Params)


def test_lstm_step_raises_on_none_state():
  """Verifies step raises ValueError if prev_state is None."""
  graph = bx.Graph('root')
  lstm = bx.LSTM(graph.child('lstm'), hidden_size=10)
  inputs = jnp.ones((1, 4))
  params = bx.Params(seed=0)

  with pytest.raises(
    ValueError, match='The LSTM step method requires a valid prev_state.'
  ):
    lstm.step(params, inputs, None)


def test_lstm_static_vs_dynamic():
  """Ensures loop (static) and scan (dynamic) produce identical results."""
  graph = bx.Graph('root')
  # Use one module instance.
  lstm = bx.LSTM(graph.child('rnn'), hidden_size=5, is_static=False)

  inputs = jnp.ones((2, 10, 4))
  params = bx.Params(seed=42)

  # Initialization pass.
  state, params = lstm.initial_state(params, inputs)
  ((_, _), params) = lstm(params, inputs, state)
  params = params.finalize()

  # Run dynamic.
  lstm.is_static = False
  ((y_dyn, _), _) = lstm(params, inputs, state)

  # Run static.
  lstm.is_static = True
  ((y_stat, _), _) = lstm(params, inputs, state)

  assert jnp.allclose(y_dyn, y_stat, atol=1e-5)


def test_lstm_reset_logic():
  """Verifies that is_reset forces the state to zero."""
  graph = bx.Graph('root')
  lstm = bx.LSTM(graph.child('lstm'), hidden_size=5)

  # Batch=1, Time=3
  inputs = jnp.ones((1, 3, 2))
  params = bx.Params(seed=0)

  # Initialization pass.
  initial_state, params = lstm.initial_state(params, inputs)
  ((_, _), params) = lstm(params, inputs, initial_state)
  params = params.finalize()

  # Reset at t=1.
  reset = jnp.array([[False, True, False]])

  ((out, _), params) = lstm(params, inputs, initial_state, is_reset=reset)

  out_0 = out[0, 0]
  out_1 = out[0, 1]
  out_2 = out[0, 2]

  # The output at t=1 (reset) should be very similar to t=0 (initial).
  assert jnp.allclose(out_0, out_1, atol=1e-5)

  # t=2 accumulates state, so it should differ from t=0.
  assert not jnp.allclose(out_0, out_2, atol=1e-5)
