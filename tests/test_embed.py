"""Tests for the Embed layer."""

import jax
import jax.numpy as jnp
import blox as bx


def test_embed_shapes():
  """Verifies embedding lookup output shapes."""
  graph = bx.Graph('root')
  embed = bx.Embed(graph.child('embed'), num_embeddings=100, embedding_size=32)

  indices = jnp.array([0, 5, 10, 50])
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  embeddings, params = embed(params, indices)

  assert embeddings.shape == (
    4,
    32,
  ), 'Output shape should be (batch, embedding_size).'


def test_embed_batched():
  """Verifies embedding lookup with batched indices."""
  graph = bx.Graph('root')
  embed = bx.Embed(graph.child('embed'), num_embeddings=100, embedding_size=16)

  # [batch, seq_len]
  indices = jnp.array([[0, 1, 2], [10, 20, 30]])
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  embeddings, params = embed(params, indices)

  assert embeddings.shape == (
    2,
    3,
    16,
  ), 'Output shape should be (batch, seq, embedding_size).'


def test_embed_attend_shapes():
  """Verifies attend (weight-tied projection) output shapes."""
  graph = bx.Graph('root')
  embed = bx.Embed(graph.child('embed'), num_embeddings=100, embedding_size=32)

  # Initialize with forward pass.
  indices = jnp.array([0, 1, 2])
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))
  _, params = embed(params, indices)
  params = params.finalize()

  # Apply attend (transpose projection).
  hidden = jnp.ones((4, 32))
  logits, _ = embed.attend(params, hidden)

  assert logits.shape == (4, 100), 'Logits shape should be (batch, vocab_size).'


def test_embed_weight_tying():
  """Verifies that attend uses the same weights as the embedding lookup."""
  graph = bx.Graph('root')
  embed = bx.Embed(graph.child('embed'), num_embeddings=10, embedding_size=4)

  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))

  # Initialize by looking up one index.
  indices = jnp.array([0])
  embeddings, params = embed(params, indices)
  params = params.finalize()

  # Get the embedding matrix directly.
  embedding_matrix = params._data[('root', 'embed', 'embedding')].value

  # Test that attend computes input @ embedding_matrix.T
  test_input = jnp.ones((2, 4))
  logits, _ = embed.attend(params, test_input)

  expected = test_input @ embedding_matrix.T
  assert jnp.allclose(
    logits, expected
  ), 'attend should compute input @ embedding.T'


def test_embed_learning():
  """Verifies gradients propagate through embedding layer."""
  graph = bx.Graph('root')
  embed = bx.Embed(graph.child('embed'), num_embeddings=10, embedding_size=4)

  indices = jnp.array([0, 1, 2])
  target = jnp.ones((3, 4)) * 5.0
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=42))

  # Initialize.
  _, params = embed(params, indices)
  params = params.finalize()

  initial_embeddings, _ = embed(params, indices)
  initial_loss = jnp.mean((initial_embeddings - target) ** 2)

  @jax.jit
  def step(p):
    trainable, non_trainable = p.split()

    def loss(t):
      full_params = t.merge(non_trainable)
      embeddings, _ = embed(full_params, indices)
      return jnp.mean((embeddings - target) ** 2)

    grads = jax.grad(loss)(trainable)
    new_trainable = jax.tree.map(lambda w, g: w - 0.1 * g, trainable, grads)
    return new_trainable.merge(non_trainable)

  # Train.
  curr = params
  for _ in range(100):
    curr = step(curr)

  final_embeddings, _ = embed(curr, indices)
  final_loss = jnp.mean((final_embeddings - target) ** 2)

  # Loss should decrease significantly.
  assert (
    final_loss < initial_loss * 0.1
  ), 'Embedding should learn (loss should decrease).'


def test_embed_parameter_path():
  """Verifies the embedding parameter is stored at the correct path."""
  graph = bx.Graph('root')
  embed = bx.Embed(graph.child('embed'), num_embeddings=50, embedding_size=8)

  indices = jnp.array([0])
  params = bx.Params(rng=bx.Rng(graph.child('rng'), seed=0))

  _, params = embed(params, indices)
  params = params.finalize()

  assert ('root', 'embed', 'embedding') in params._data
  assert params._data[('root', 'embed', 'embedding')].value.shape == (50, 8)
