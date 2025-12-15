"""Standard neural network blocks.

This module contains implementations of common layers like Linear and LSTM.
It serves as the user-facing library of pre-built components.
"""

from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp

from . import interfaces as bx

Initializer = jax.nn.initializers.Initializer
PaddingLike = str | Sequence[tuple[int, int]]


class Embed(bx.Module):
  """Embedding layer that maps integer indices to dense vectors.

  Supports weight tying for language models via the `attend` method, which
  applies the transpose of the embedding matrix (useful for output projections).

  Example:
    embed = Embed(graph / 'embed', num_embeddings=10000, embedding_size=512)

    # Forward pass: indices -> embeddings
    embeddings, params = embed(params, token_ids)

    # Weight-tied output projection: hidden -> logits
    logits, params = embed.attend(params, hidden_states)
  """

  def __init__(
    self,
    graph: bx.Graph,
    num_embeddings: int,
    embedding_size: int,
    embedding_init: Initializer | None = None,
  ) -> None:
    """Initializes the Embed module.

    Args:
      graph: The graph node for this module.
      num_embeddings: Size of the vocabulary (number of unique tokens).
      embedding_size: Dimensionality of the embedding vectors.
      embedding_init: Initializer for the embedding matrix. Defaults to normal.
    """
    super().__init__(graph)
    self.num_embeddings = num_embeddings
    self.embedding_size = embedding_size
    if embedding_init is not None:
      self.embedding_init = embedding_init
    else:
      self.embedding_init = jax.nn.initializers.variance_scaling(
        1.0, 'fan_in', 'normal', out_axis=0
      )

  def __call__(
    self,
    params: bx.Params,
    indices: jax.Array,
  ) -> tuple[jax.Array, bx.Params]:
    """Looks up embeddings for the given indices.

    Args:
      params: The parameters container.
      indices: Integer array of token indices. Shape [...].

    Returns:
      A tuple (embeddings, params). Embeddings have shape [..., embedding_size].
    """
    embedding_matrix, params = self.get_param(
      params,
      'embedding',
      (self.num_embeddings, self.embedding_size),
      self.embedding_init,
    )
    return embedding_matrix[indices], params

  def attend(
    self,
    params: bx.Params,
    inputs: jax.Array,
  ) -> tuple[jax.Array, bx.Params]:
    """Applies the transpose of the embedding matrix (for weight tying).

    This is commonly used in language models where the output projection
    shares weights with the input embedding.

    Args:
      params: The parameters container.
      inputs: Input array of shape [..., embedding_size].

    Returns:
      A tuple (logits, params). Logits have shape [..., num_embeddings].
    """
    embedding_matrix, params = self.get_param(
      params,
      'embedding',
      (self.num_embeddings, self.embedding_size),
      self.embedding_init,
    )
    # inputs @ embedding_matrix.T
    return jnp.dot(inputs, embedding_matrix.T), params


class Linear(bx.Module):
  """A standard linear transformation layer.

  Computes `output = input @ kernel + bias`.

  Supports model parallelism via metadata. Example for sharding weights:
    linear = Linear(
      graph.child('linear'),
      output_size=1024,
      kernel_metadata={'sharding': (None, 'model')},  # Shard output dim
      bias_metadata={'sharding': ('model',)},
    )
  """

  def __init__(
    self,
    graph: bx.Graph,
    output_size: int,
    use_bias: bool = True,
    kernel_init: Initializer | None = None,
    bias_init: Initializer | None = None,
    kernel_metadata: dict[str, Any] | None = None,
    bias_metadata: dict[str, Any] | None = None,
  ) -> None:
    """Initializes the Linear module.

    Args:
      graph: The graph node for this module.
      output_size: The dimensionality of the output features.
      use_bias: Whether to add a learnable bias vector.
      kernel_init: Initializer for the weight matrix. Defaults to Lecun Normal.
      bias_init: Initializer for the bias vector. Defaults to Zeros.
      kernel_metadata: Optional metadata for the kernel parameter. Common keys:
        - 'sharding': tuple like (None, 'model') for model parallelism
      bias_metadata: Optional metadata for the bias parameter. Common keys:
        - 'sharding': tuple like ('model',) for model parallelism
    """
    super().__init__(graph)
    self.output_size = output_size
    self.use_bias = use_bias
    self.kernel_init = kernel_init or jax.nn.initializers.lecun_normal()
    self.bias_init = bias_init or jax.nn.initializers.zeros
    self.kernel_metadata = kernel_metadata
    self.bias_metadata = bias_metadata

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
    kernel, params = self.get_param(
      params,
      'kernel',
      (input_size, self.output_size),
      self.kernel_init,
      metadata=self.kernel_metadata,
    )
    outputs = jnp.dot(inputs, kernel, precision=precision)

    if self.use_bias:
      bias, params = self.get_param(
        params,
        'bias',
        (self.output_size,),
        self.bias_init,
        metadata=self.bias_metadata,
      )
      bias = jnp.broadcast_to(bias, outputs.shape)
      outputs = outputs + bias

    return outputs, params


class LSTMState(NamedTuple):
  """Holds the hidden and cell states for an LSTM."""

  hidden: jax.Array
  cell: jax.Array


class LSTM(bx.RecurrenceBase[jax.Array, LSTMState, jax.Array, jax.Array]):
  """Long Short-Term Memory (LSTM) Recurrent Neural Network.

  This module implements a standard LSTM cell. It inherits from RecurrenceBase,
  automatically providing support for both single-step execution (`__call__`)
  and efficient sequence processing (`apply` with scanning).

  Example:
    lstm = LSTM(graph.child('lstm'), hidden_size=128)

    # Initialize state first:
    state, params = lstm.initial_state(params, inputs)

    # Single-step processing (e.g., for interactive use):
    (outputs, state), params = lstm(params, inputs, state)

    # Sequence processing (uses jax.lax.scan for efficiency):
    (outputs, state), params = lstm.apply(params, inputs_sequence, state)
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

  def __call__(
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
      A nested tuple ((output, new_state), params).
      The output of the LSTM is the hidden state.

    Raises:
      ValueError: If prev_state is None.
    """
    del is_training  # Currently unused.
    if prev_state is None:
      raise ValueError('The LSTM __call__ method requires a valid prev_state.')

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


class Dropout(bx.Module):
  """Dropout layer for regularization.

  During training, randomly zeros elements with probability `rate` and scales
  the remaining elements by `1 / (1 - rate)` to maintain expected values.
  During inference, this layer is a no-op.

  Can optionally use its own Rng module instead of params.next_key() for
  finer control over randomness streams.

  For shard_map/vmap usage where different devices need different dropout masks,
  call `params.fold_in_axes('axis_name')` at the start and `fold_out_axes()` at
  the end:

    @jax.vmap(..., axis_name='batch')
    def apply(params, x):
      params = params.fold_in_axes('batch')
      out, params = dropout(params, x, is_training=True)
      return out, params.fold_out_axes('batch')
  """

  def __init__(
    self,
    graph: bx.Graph,
    rate: float = 0.5,
    rng: bx.Rng | None = None,
  ) -> None:
    """Initializes the Dropout module.

    Args:
      graph: The graph node for this module.
      rate: The probability of dropping each element (0.0 to 1.0).
      rng: Optional Rng module for this layer. If provided, uses this instead
        of params.next_key() for generating dropout masks.
    """
    super().__init__(graph)
    if not 0.0 <= rate < 1.0:
      raise ValueError(f'Dropout rate must be in [0.0, 1.0), got {rate}.')
    self.rate = rate
    self.rng = rng

  def __call__(
    self,
    params: bx.Params,
    inputs: jax.Array,
    is_training: bool = True,
  ) -> tuple[jax.Array, bx.Params]:
    """Applies dropout to the inputs.

    Args:
      params: The parameters container.
      inputs: The input array.
      is_training: If True, applies dropout. If False, returns inputs unchanged.

    Returns:
      A tuple (output, params).
    """
    if not is_training or self.rate == 0.0:
      return inputs, params

    # Use module-owned RNG if provided, otherwise params.next_key().
    if self.rng is not None:
      key, params = self.rng(params)
    else:
      key, params = params.next_key()

    keep_rate = 1.0 - self.rate
    mask = jax.random.bernoulli(key, keep_rate, inputs.shape)
    return inputs * mask / keep_rate, params


class LayerNorm(bx.Module):
  """Layer Normalization.

  Normalizes over the last axis (features) of the input. Supports cross-device
  statistics aggregation via axis_name for use with jax.shard_map.
  """

  def __init__(
    self,
    graph: bx.Graph,
    epsilon: float = 1e-5,
    use_scale: bool = True,
    use_bias: bool = True,
    scale_init: Initializer | None = None,
    bias_init: Initializer | None = None,
    axis_name: str | None = None,
    axis_index_groups: Sequence[Sequence[int]] | None = None,
  ) -> None:
    """Initializes the LayerNorm module.

    Args:
      graph: The graph node for this module.
      epsilon: Small constant for numerical stability.
      use_scale: Whether to use a learnable scale parameter.
      use_bias: Whether to use a learnable bias parameter.
      scale_init: Initializer for scale. Defaults to ones.
      bias_init: Initializer for bias. Defaults to zeros.
      axis_name: The axis name used to combine statistics from multiple devices.
        See jax.shard_map for a description of axis names.
      axis_index_groups: Groups of axis indices within that named axis
        representing subsets of devices to reduce over. For example,
        [[0, 1], [2, 3]] would independently normalize over the first two and
        last two devices. See jax.lax.psum for more details.
    """
    super().__init__(graph)
    self.epsilon = epsilon
    self.use_scale = use_scale
    self.use_bias = use_bias
    self.scale_init = scale_init or jax.nn.initializers.ones
    self.bias_init = bias_init or jax.nn.initializers.zeros
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups

  def __call__(
    self,
    params: bx.Params,
    inputs: jax.Array,
  ) -> tuple[jax.Array, bx.Params]:
    """Applies layer normalization.

    Args:
      params: The parameters container.
      inputs: The input array with shape [..., features].

    Returns:
      A tuple (normalized_output, params).
    """
    features = inputs.shape[-1]

    # Compute mean and variance over last axis.
    mean = jnp.mean(inputs, axis=-1, keepdims=True)
    var = jnp.var(inputs, axis=-1, keepdims=True)

    # Cross-device aggregation if axis_name is provided.
    if self.axis_name is not None:
      mean = jax.lax.pmean(
        mean, self.axis_name, axis_index_groups=self.axis_index_groups
      )
      var = jax.lax.pmean(
        var, self.axis_name, axis_index_groups=self.axis_index_groups
      )

    # Normalize.
    normalized = (inputs - mean) / jnp.sqrt(var + self.epsilon)

    # Scale and shift.
    if self.use_scale:
      scale, params = self.get_param(
        params, 'scale', (features,), self.scale_init
      )
      normalized = normalized * scale

    if self.use_bias:
      bias, params = self.get_param(params, 'bias', (features,), self.bias_init)
      normalized = normalized + bias

    return normalized, params


class RMSNorm(bx.Module):
  """Root Mean Square Layer Normalization.

  Normalizes using only the RMS of the input (no mean subtraction).
  This is computationally simpler than LayerNorm. Supports cross-device
  statistics aggregation via axis_name for use with jax.shard_map.
  """

  def __init__(
    self,
    graph: bx.Graph,
    epsilon: float = 1e-5,
    use_scale: bool = True,
    scale_init: Initializer | None = None,
    axis_name: str | None = None,
    axis_index_groups: Sequence[Sequence[int]] | None = None,
  ) -> None:
    """Initializes the RMSNorm module.

    Args:
      graph: The graph node for this module.
      epsilon: Small constant for numerical stability.
      use_scale: Whether to use a learnable scale parameter.
      scale_init: Initializer for scale. Defaults to ones.
      axis_name: The axis name used to combine statistics from multiple devices.
        See jax.shard_map for a description of axis names.
      axis_index_groups: Groups of axis indices within that named axis
        representing subsets of devices to reduce over. For example,
        [[0, 1], [2, 3]] would independently normalize over the first two and
        last two devices. See jax.lax.psum for more details.
    """
    super().__init__(graph)
    self.epsilon = epsilon
    self.use_scale = use_scale
    self.scale_init = scale_init or jax.nn.initializers.ones
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups

  def __call__(
    self,
    params: bx.Params,
    inputs: jax.Array,
  ) -> tuple[jax.Array, bx.Params]:
    """Applies RMS normalization.

    Args:
      params: The parameters container.
      inputs: The input array with shape [..., features].

    Returns:
      A tuple (normalized_output, params).
    """
    features = inputs.shape[-1]

    # Compute mean of squares.
    mean_sq = jnp.mean(inputs**2, axis=-1, keepdims=True)

    # Cross-device aggregation if axis_name is provided.
    if self.axis_name is not None:
      mean_sq = jax.lax.pmean(
        mean_sq, self.axis_name, axis_index_groups=self.axis_index_groups
      )

    # Compute RMS.
    rms = jnp.sqrt(mean_sq + self.epsilon)

    # Normalize.
    normalized = inputs / rms

    # Scale.
    if self.use_scale:
      scale, params = self.get_param(
        params, 'scale', (features,), self.scale_init
      )
      normalized = normalized * scale

    return normalized, params


class BatchNorm(bx.Module):
  """Batch Normalization.

  Normalizes over the batch and spatial dimensions, maintaining running
  statistics for inference. During training, computes batch statistics and
  updates the running averages. During inference, uses the stored running
  statistics.

  The input is expected to have shape (batch, *spatial_dims, features).
  Normalization is applied over all axes except the last (features) axis.

  Example:
    bn = BatchNorm(graph.child('bn'))

    # Training: uses batch statistics and updates running stats.
    y, params = bn(params, x, is_training=True)

    # Inference: uses running statistics.
    y, params = bn(params, x, is_training=False)
  """

  def __init__(
    self,
    graph: bx.Graph,
    momentum: float = 0.9,
    epsilon: float = 1e-5,
    use_scale: bool = True,
    use_bias: bool = True,
    scale_init: Initializer | None = None,
    bias_init: Initializer | None = None,
    axis_name: str | None = None,
    axis_index_groups: Sequence[Sequence[int]] | None = None,
  ) -> None:
    """Initializes the BatchNorm module.

    Args:
      graph: The graph node for this module.
      momentum: Momentum for the exponential moving average of running stats.
        A value of 0.9 means running_stat = 0.9 * running_stat + 0.1 * batch_stat.
      epsilon: Small constant for numerical stability.
      use_scale: Whether to use a learnable scale parameter (gamma).
      use_bias: Whether to use a learnable bias parameter (beta).
      scale_init: Initializer for scale. Defaults to ones.
      bias_init: Initializer for bias. Defaults to zeros.
      axis_name: The axis name used to combine statistics from multiple devices.
        See jax.shard_map for a description of axis names.
      axis_index_groups: Groups of axis indices within that named axis
        representing subsets of devices to reduce over.
    """
    super().__init__(graph)
    self.momentum = momentum
    self.epsilon = epsilon
    self.use_scale = use_scale
    self.use_bias = use_bias
    self.scale_init = scale_init or jax.nn.initializers.ones
    self.bias_init = bias_init or jax.nn.initializers.zeros
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups

  def __call__(
    self,
    params: bx.Params,
    inputs: jax.Array,
    is_training: bool = True,
  ) -> tuple[jax.Array, bx.Params]:
    """Applies batch normalization.

    Args:
      params: The parameters container.
      inputs: The input array with shape (batch, *spatial_dims, features).
      is_training: If True, uses batch statistics and updates running stats.
        If False, uses stored running statistics (inference).

    Returns:
      A tuple (normalized_output, params).
    """
    features = inputs.shape[-1]
    # Axes to reduce over: all except the last (features) axis.
    reduce_axes = tuple(range(inputs.ndim - 1))

    # Get or create running statistics (non-trainable).
    running_mean, params = self.get_param(
      params=params,
      name='running_mean',
      shape=(features,),
      init=jax.nn.initializers.zeros,
      trainable=False,
    )
    running_var, params = self.get_param(
      params=params,
      name='running_var',
      shape=(features,),
      init=jax.nn.initializers.ones,
      trainable=False,
    )

    if is_training:
      # Compute batch statistics.
      mean = jnp.mean(inputs, axis=reduce_axes)
      var = jnp.var(inputs, axis=reduce_axes)

      # Cross-device aggregation if axis_name is provided.
      if self.axis_name is not None:
        mean = jax.lax.pmean(
          mean, self.axis_name, axis_index_groups=self.axis_index_groups
        )
        var = jax.lax.pmean(
          var, self.axis_name, axis_index_groups=self.axis_index_groups
        )

      # Update running statistics with exponential moving average.
      # stop_gradient prevents backprop through running stats.
      new_running_mean = self.momentum * running_mean + (
        1 - self.momentum
      ) * jax.lax.stop_gradient(mean)
      new_running_var = self.momentum * running_var + (
        1 - self.momentum
      ) * jax.lax.stop_gradient(var)

      # Store updated running statistics.
      params = self.set_param(
        params=params, name='running_mean', value=new_running_mean
      )
      params = self.set_param(
        params=params, name='running_var', value=new_running_var
      )
    else:
      # Use running statistics for inference.
      mean = running_mean
      var = running_var

    # Normalize.
    normalized = (inputs - mean) / jnp.sqrt(var + self.epsilon)

    # Scale and shift.
    if self.use_scale:
      scale, params = self.get_param(
        params, 'scale', (features,), self.scale_init
      )
      normalized = normalized * scale

    if self.use_bias:
      bias, params = self.get_param(params, 'bias', (features,), self.bias_init)
      normalized = normalized + bias

    return normalized, params


def _normalize_tuple(x: int | Sequence[int], n: int) -> tuple[int, ...]:
  """Converts int or sequence to a tuple of length n."""
  if isinstance(x, int):
    return (x,) * n
  return tuple(x)


class Conv(bx.Module):
  """General N-dimensional convolution layer.

  Supports 1D, 2D, and 3D convolutions based on the length of kernel_size.
  Uses channels-last convention: (batch, *spatial_dims, channels).

  Example:
    # 2D convolution for images (NHWC format)
    conv = Conv(graph.child('conv'), output_channels=64, kernel_size=(3, 3))
    y, params = conv(params, x)  # x: [batch, height, width, channels]

    # 1D convolution for sequences (NLC format)
    conv = Conv(graph.child('conv'), output_channels=64, kernel_size=3)
    y, params = conv(params, x)  # x: [batch, length, channels]
  """

  def __init__(
    self,
    graph: bx.Graph,
    output_channels: int,
    kernel_size: int | Sequence[int],
    strides: int | Sequence[int] = 1,
    padding: PaddingLike = 'SAME',
    input_dilation: int | Sequence[int] = 1,
    kernel_dilation: int | Sequence[int] = 1,
    feature_group_count: int = 1,
    use_bias: bool = True,
    kernel_init: Initializer | None = None,
    bias_init: Initializer | None = None,
    kernel_metadata: dict[str, Any] | None = None,
    bias_metadata: dict[str, Any] | None = None,
  ) -> None:
    """Initializes the Conv module.

    Args:
      graph: The graph node for this module.
      output_channels: Number of output channels.
      kernel_size: Size of the convolutional kernel. An int is broadcast to
        all spatial dimensions.
      strides: Stride of the convolution. An int is broadcast to all spatial
        dimensions.
      padding: Padding mode. Either 'SAME', 'VALID', or a sequence of
        (low, high) padding pairs for each spatial dimension.
      input_dilation: Dilation of the input (transposed convolution).
      kernel_dilation: Dilation of the kernel (atrous convolution).
      feature_group_count: Number of feature groups for grouped convolution.
        Set to input_channels for depthwise convolution.
      use_bias: Whether to add a learnable bias.
      kernel_init: Initializer for the kernel. Defaults to Lecun Normal.
      bias_init: Initializer for the bias. Defaults to zeros.
      kernel_metadata: Optional metadata for the kernel parameter.
      bias_metadata: Optional metadata for the bias parameter.
    """
    super().__init__(graph)
    self.output_channels = output_channels
    self.kernel_size = (
      (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
    )
    self.strides = strides
    self.padding = padding
    self.input_dilation = input_dilation
    self.kernel_dilation = kernel_dilation
    self.feature_group_count = feature_group_count
    self.use_bias = use_bias
    self.kernel_init = kernel_init or jax.nn.initializers.lecun_normal()
    self.kernel_metadata = kernel_metadata
    self.bias_metadata = bias_metadata
    self.bias_init = bias_init or jax.nn.initializers.zeros

  def __call__(
    self,
    params: bx.Params,
    inputs: jax.Array,
    precision: jax.lax.Precision | None = None,
  ) -> tuple[jax.Array, bx.Params]:
    """Applies the convolution.

    Args:
      params: The parameters container.
      inputs: Input array with shape (batch, *spatial_dims, input_channels).
      precision: Optional precision for the convolution.

    Returns:
      A tuple (output, params). Output has shape (batch, *out_spatial, output_channels).

    Raises:
      ValueError: If input rank doesn't match kernel dimensions.
    """
    num_spatial = len(self.kernel_size)
    expected_rank = num_spatial + 2  # batch + spatial + channels

    if inputs.ndim != expected_rank:
      raise ValueError(
        f'Expected input rank {expected_rank} for {num_spatial}D conv, '
        f'got {inputs.ndim}.'
      )

    input_channels = inputs.shape[-1]

    if input_channels % self.feature_group_count != 0:
      raise ValueError(
        f'input_channels ({input_channels}) must be divisible by '
        f'feature_group_count ({self.feature_group_count}).'
      )

    # Kernel shape: (*kernel_size, input_channels // groups, output_channels)
    kernel_shape = self.kernel_size + (
      input_channels // self.feature_group_count,
      self.output_channels,
    )
    kernel, params = self.get_param(
      params,
      'kernel',
      kernel_shape,
      self.kernel_init,
      metadata=self.kernel_metadata,
    )

    # Normalize strides and dilations to tuples.
    strides = _normalize_tuple(self.strides, num_spatial)
    input_dilation = _normalize_tuple(self.input_dilation, num_spatial)
    kernel_dilation = _normalize_tuple(self.kernel_dilation, num_spatial)

    # Build dimension numbers for channels-last format.
    # Input: (N, *spatial, C) -> lax expects (N, C, *spatial)
    # We use dimension_numbers to avoid transposing.
    spatial_dims = tuple(range(1, num_spatial + 1))
    lhs_spec = (0, num_spatial + 1) + spatial_dims  # (N, C, *spatial)
    rhs_spec = (num_spatial + 1, num_spatial) + tuple(range(num_spatial))
    out_spec = (0, num_spatial + 1) + spatial_dims
    dimension_numbers = jax.lax.ConvDimensionNumbers(
      lhs_spec=lhs_spec, rhs_spec=rhs_spec, out_spec=out_spec
    )

    # Apply convolution.
    outputs = jax.lax.conv_general_dilated(
      inputs,
      kernel,
      window_strides=strides,
      padding=self.padding,
      lhs_dilation=input_dilation,
      rhs_dilation=kernel_dilation,
      dimension_numbers=dimension_numbers,
      feature_group_count=self.feature_group_count,
      precision=precision,
    )

    # Add bias.
    if self.use_bias:
      bias, params = self.get_param(
        params,
        'bias',
        (self.output_channels,),
        self.bias_init,
        metadata=self.bias_metadata,
      )
      outputs = outputs + bias

    return outputs, params


def max_pool(
  inputs: jax.Array,
  window_shape: int | Sequence[int],
  strides: int | Sequence[int] | None = None,
  padding: str = 'VALID',
) -> jax.Array:
  """Applies max pooling over spatial dimensions.

  Uses channels-last convention: (batch, *spatial_dims, channels).

  Args:
    inputs: Input array with shape (batch, *spatial_dims, channels).
    window_shape: Size of the pooling window. An int is broadcast to all
      spatial dimensions.
    strides: Stride of the pooling. If None, uses window_shape (non-overlapping).
    padding: Padding mode. Either 'SAME' or 'VALID'.

  Returns:
    Pooled output array.

  Example:
    # 2x2 max pooling with stride 2
    y = max_pool(x, window_shape=2, strides=2)
  """
  num_spatial = inputs.ndim - 2
  window = _normalize_tuple(
    window_shape if isinstance(window_shape, int) else tuple(window_shape),
    num_spatial,
  )
  strides_tuple = (
    window if strides is None else _normalize_tuple(strides, num_spatial)
  )

  # jax.lax.reduce_window expects window and strides for all dims.
  # Format: (batch, *spatial, channels)
  full_window = (1,) + window + (1,)
  full_strides = (1,) + strides_tuple + (1,)

  return jax.lax.reduce_window(
    inputs,
    init_value=-jnp.inf,
    computation=jax.lax.max,
    window_dimensions=full_window,
    window_strides=full_strides,
    padding=padding,
  )


def avg_pool(
  inputs: jax.Array,
  window_shape: int | Sequence[int],
  strides: int | Sequence[int] | None = None,
  padding: str = 'VALID',
) -> jax.Array:
  """Applies average pooling over spatial dimensions.

  Uses channels-last convention: (batch, *spatial_dims, channels).

  Args:
    inputs: Input array with shape (batch, *spatial_dims, channels).
    window_shape: Size of the pooling window. An int is broadcast to all
      spatial dimensions.
    strides: Stride of the pooling. If None, uses window_shape (non-overlapping).
    padding: Padding mode. Either 'SAME' or 'VALID'.

  Returns:
    Pooled output array.

  Example:
    # 2x2 average pooling with stride 2
    y = avg_pool(x, window_shape=2, strides=2)
  """
  num_spatial = inputs.ndim - 2
  window = _normalize_tuple(
    window_shape if isinstance(window_shape, int) else tuple(window_shape),
    num_spatial,
  )
  strides_tuple = (
    window if strides is None else _normalize_tuple(strides, num_spatial)
  )

  # jax.lax.reduce_window expects window and strides for all dims.
  full_window = (1,) + window + (1,)
  full_strides = (1,) + strides_tuple + (1,)

  # Sum pooling.
  pooled_sum = jax.lax.reduce_window(
    inputs,
    init_value=0.0,
    computation=jax.lax.add,
    window_dimensions=full_window,
    window_strides=full_strides,
    padding=padding,
  )

  # Divide by window size for average.
  window_size = 1
  for w in window:
    window_size *= w

  return pooled_sum / window_size
