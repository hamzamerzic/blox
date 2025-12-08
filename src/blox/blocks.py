"""
Standard neural network blocks.

This module contains implementations of common layers like Linear and LSTM.
It serves as the user-facing library of pre-built components.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import interfaces as bx

Initializer = jax.nn.initializers.Initializer


class Linear(bx.Module):
  """A standard linear transformation layer.

  Computes `output = input @ w + b`.
  """

  def __init__(
    self,
    graph: bx.Graph,
    output_size: int,
    with_bias: bool = True,
    w_init: Initializer | None = None,
    b_init: Initializer | None = None,
  ) -> None:
    """Initializes the Linear module.

    Args:
      graph: The graph node for this module.
      output_size: The dimensionality of the output features.
      with_bias: Whether to add a learnable bias vector.
      w_init: Initializer for the weight matrix. Defaults to Lecun Normal.
      b_init: Initializer for the bias vector. Defaults to Zeros.
    """
    super().__init__(graph)
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init or jax.nn.initializers.lecun_normal()
    self.b_init = b_init or jax.nn.initializers.constant(0.0)

  def __call__(
    self,
    params: bx.Params,
    inputs: jax.Array,
    precision: jax.lax.Precision | None = None,
  ) -> tuple[jax.Array, bx.Params]:
    """Applies the linear transformation.

    Args:
      params: The parameters container.
      inputs: The input array. Must have at least one dimension.
         Shape should be [..., input_features].
      precision: Optional precision for the matrix multiplication.

    Returns:
      A tuple (output, params). The output has shape [..., output_size].

    Raises:
      ValueError: If the input is a scalar (rank 0).
    """
    if not inputs.shape:
      raise ValueError('Input must not be scalar.')

    input_size = inputs.shape[-1]
    w, params = self.get_param(
      params, 'w', (input_size, self.output_size), self.w_init
    )
    outputs = jnp.dot(inputs, w, precision=precision)

    if self.with_bias:
      b, params = self.get_param(params, 'b', (self.output_size,), self.b_init)
      b = jnp.broadcast_to(b, outputs.shape)
      outputs = outputs + b

    return outputs, params


class LSTMState(NamedTuple):
  """Holds the hidden and cell states for an LSTM."""

  hidden: jax.Array
  cell: jax.Array


class LSTM(bx.RNNCore[jax.Array, LSTMState, jax.Array, jax.Array]):
  """Long Short-Term Memory (LSTM) Recurrent Neural Network.

  This module implements a standard LSTM cell. It inherits from RNNCore,
  automatically providing support for both step-by-step execution (`step`)
  and efficient sequence compilation (`__call__` / `scan`).
  """

  def __init__(
    self,
    graph: bx.Graph,
    hidden_size: int,
    is_static: bool = False,
  ) -> None:
    """Initializes the LSTM.

    Args:
      graph: The graph node for this module.
      hidden_size: The dimensionality of the hidden and cell states.
      is_static: If True, uses Python loops for sequence processing.
                 If False, uses jax.lax.scan (default).
    """
    super().__init__(graph, is_static)
    self.hidden_size = hidden_size
    # We use a single Linear layer to project inputs to the 4 gates (i, g, f, o).
    self.gates = Linear(graph.child('gates'), output_size=4 * hidden_size)

  def initial_state(
    self, params: bx.Params, inputs: jax.Array
  ) -> tuple[LSTMState, bx.Params]:
    """Creates the initial zero state.

    Args:
      params: The parameters container.
      inputs: The input array, used to infer the batch size (dimension 0).

    Returns:
      A tuple (LSTMState, params), where both hidden and cell states are zeros.
    """
    batch_size = inputs.shape[0]
    return LSTMState(
      hidden=jnp.zeros((batch_size, self.hidden_size)),
      cell=jnp.zeros((batch_size, self.hidden_size)),
    ), params

  def step(
    self,
    params: bx.Params,
    inputs: jax.Array,
    prev_state: LSTMState | None,
    is_reset: jax.Array | None = None,
    is_training: bool = True,
  ) -> tuple[tuple[jax.Array, LSTMState], bx.Params]:
    """Computes a single step of the LSTM recurrence.

    Args:
      params: The parameters container.
      inputs: The input at the current time step. Shape [Batch, input_size].
      prev_state: The previous LSTM state. Must not be None.
      is_reset: Optional boolean array [Batch]. If True for a batch element,
        the state is reset to zero *before* computing the step.
      is_training: Unused.

    Returns:
      A nested tuple ((hidden_state, new_state), params).
      The output of the LSTM is the hidden state.

    Raises:
      ValueError: If prev_state is None.
    """
    del is_training  # Currently unused.
    if prev_state is None:
      raise ValueError('The LSTM step method requires a valid prev_state.')

    # Apply reset mask if provided.
    prev_state = self.maybe_reset_state(params, prev_state, inputs, is_reset)
    prev_h, prev_c = prev_state.hidden, prev_state.cell

    # Concatenate input and previous hidden state.
    x_and_h = jnp.concatenate([inputs, prev_h], axis=-1)

    # Project to gates.
    gated, params = self.gates(params, x_and_h)

    # Split into input, gate, forget, and output components.
    i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)

    # Apply activations.
    f = jax.nn.sigmoid(f)
    c = f * prev_c + jax.nn.sigmoid(i) * jnp.tanh(g)
    h = jax.nn.sigmoid(o) * jnp.tanh(c)

    new_state = LSTMState(hidden=h, cell=c)

    # Output is h, state is (h, c).
    return (h, new_state), params
