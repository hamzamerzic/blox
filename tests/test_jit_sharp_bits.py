"""Tests documenting JIT behavior with module modification.

These tests document a known sharp bit: when modules are captured in closures
by JAX's JIT, modifying module attributes after the first call won't trigger
recompilation. The cached version continues to use the old attribute values.

See: https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html

Key patterns:
1. Apply modifications BEFORE the first JIT call
2. Clear the JIT cache after modifications (jit_fn.clear_cache())
3. Pass mutable state as Params, not module attributes
"""

import jax
import jax.numpy as jnp

import blox as bx


def test_module_modification_before_jit_works():
  """Verifies that module modifications before JIT are captured correctly."""
  graph = bx.Graph('net')
  rng = bx.Rng(graph.child('rng'))
  lstm = bx.LSTM(graph.child('lstm'), hidden_size=8, rng=rng)

  # Modify BEFORE first JIT call.
  lstm.is_static = True

  x = jnp.ones((2, 5, 4))
  params = rng.seed(bx.Params(), seed=42)

  # Initialize params (unlocked).
  state, params = lstm.initial_state(params, x[:, 0])
  _, params = lstm.apply(params, x, state)
  params = params.locked()

  @jax.jit
  def forward(params, x, state):
    return lstm.apply(params, x, state)

  # First call captures is_static=True.
  (out, _), _ = forward(params, x, state)

  assert out.shape == (2, 5, 8)


def test_module_modification_after_jit_not_captured():
  """Demonstrates that module modifications after JIT are NOT captured.

  This is a known JAX behavior: modules in closures are captured at first
  trace, and subsequent modifications don't trigger recompilation.
  """
  graph = bx.Graph('net')
  rng = bx.Rng(graph.child('rng'))
  lstm = bx.LSTM(graph.child('lstm'), hidden_size=8, rng=rng)

  x = jnp.ones((2, 5, 4))
  params = rng.seed(bx.Params(), seed=42)

  # Initialize params.
  state, params = lstm.initial_state(params, x[:, 0])
  _, params = lstm.apply(params, x, state)
  params = params.locked()

  # Track how many times the function is actually traced.
  trace_count = [0]

  @jax.jit
  def forward(params, x, state):
    trace_count[0] += 1
    return lstm.apply(params, x, state)

  # First call - traces the function.
  lstm.is_static = False
  (out1, _), _ = forward(params, x, state)
  assert trace_count[0] == 1

  # Second call with same shapes - uses cached version.
  (out2, _), _ = forward(params, x, state)
  assert trace_count[0] == 1  # No re-trace.

  # Modify module attribute after JIT.
  lstm.is_static = True

  # Third call - STILL uses cached version despite modification.
  (out3, _), _ = forward(params, x, state)
  assert trace_count[0] == 1  # IMPORTANT: No re-trace! Modification ignored.


def test_clear_cache_after_modification_triggers_retrace():
  """Verifies that clearing JIT cache after modification triggers retrace."""
  graph = bx.Graph('net')
  rng = bx.Rng(graph.child('rng'))
  lstm = bx.LSTM(graph.child('lstm'), hidden_size=8, rng=rng)

  x = jnp.ones((2, 5, 4))
  params = rng.seed(bx.Params(), seed=42)

  # Initialize params.
  state, params = lstm.initial_state(params, x[:, 0])
  _, params = lstm.apply(params, x, state)
  params = params.locked()

  trace_count = [0]

  @jax.jit
  def forward(params, x, state):
    trace_count[0] += 1
    return lstm.apply(params, x, state)

  # First call.
  lstm.is_static = False
  forward(params, x, state)
  assert trace_count[0] == 1

  # Modify and clear cache.
  lstm.is_static = True
  getattr(forward, 'clear_cache')()

  # Now the modification is captured.
  forward(params, x, state)
  assert trace_count[0] == 2  # Retraced after cache clear.


def test_graph_walk_modification_pattern():
  """Demonstrates the correct pattern for graph.walk() modifications.

  When using graph.walk() to modify modules (e.g., for LoRA), apply the
  modifications BEFORE any JIT calls that use those modules.
  """
  graph = bx.Graph('net')
  rng = bx.Rng(graph.child('rng'))

  # Create a simple model with multiple LSTMs.
  lstm1 = bx.LSTM(graph.child('lstm1'), hidden_size=8, rng=rng)
  lstm2 = bx.LSTM(graph.child('lstm2'), hidden_size=8, rng=rng)

  # Use graph.walk() to modify all LSTMs BEFORE JIT.
  for path, module in graph.walk():
    if isinstance(module, bx.LSTM):
      module.is_static = True

  # Verify modifications were applied.
  assert lstm1.is_static is True
  assert lstm2.is_static is True

  # Now JIT will capture the modified state.
  x = jnp.ones((2, 5, 4))
  params = rng.seed(bx.Params(), seed=42)

  # Initialize params first.
  state1, params = lstm1.initial_state(params, x[:, 0])
  (out1, _), params = lstm1.apply(params, x, state1)
  state2, params = lstm2.initial_state(params, out1[:, 0])
  (out2, _), params = lstm2.apply(params, out1, state2)
  params = params.locked()

  @jax.jit
  def forward(params, x):
    state1, params = lstm1.initial_state(params, x[:, 0])
    (out1, _), params = lstm1.apply(params, x, state1)
    state2, params = lstm2.initial_state(params, out1[:, 0])
    (out2, _), params = lstm2.apply(params, out1, state2)
    return out2, params

  out, _ = forward(params, x)
  assert out.shape == (2, 5, 8)


def test_lora_pattern_before_jit():
  """Demonstrates the LoRA pattern: modify get_param BEFORE JIT.

  LoRA works by monkey-patching get_param on layers. This must be done
  before the JIT function is first called, otherwise the original
  get_param will be cached.
  """
  graph = bx.Graph('net')
  rng = bx.Rng(graph.child('rng'))
  linear = bx.Linear(graph.child('linear'), output_size=8, rng=rng)

  # Store original get_param.
  original_get_param = linear.get_param
  call_count = [0]

  # Wrap get_param to add LoRA-like behavior.
  def lora_get_param(params, name, shape=None, init=None, **kwargs):
    call_count[0] += 1
    return original_get_param(params, name, shape, init, **kwargs)

  # Apply BEFORE JIT.
  linear.get_param = lora_get_param  # pyrefly: ignore[bad-assignment]

  x = jnp.ones((2, 4))
  params = rng.seed(bx.Params(), seed=42)

  # Initialize first.
  _, params = linear(params, x)
  params = params.locked()

  @jax.jit
  def forward(params, x):
    return linear(params, x)

  # First call - traces with our wrapped get_param.
  out, _ = forward(params, x)
  first_count = call_count[0]
  assert first_count > 0  # Our wrapper was called.

  # Second call - uses cached version.
  out, _ = forward(params, x)
  # Call count doesn't increase in cached version (no retrace).
  assert call_count[0] == first_count


def test_separate_jit_caches_for_different_configs():
  """Demonstrates creating separate JIT caches for different module configs.

  When you need both static and dynamic versions of a function, you can:
  1. Define the function without @jax.jit decorator
  2. Create jit(fn) with one module config
  3. Modify the module
  4. Create another jit(fn) with the new config

  Each jax.jit() call creates a separate cache, so both versions coexist.
  """
  graph = bx.Graph('net')
  rng = bx.Rng(graph.child('rng'))
  lstm = bx.LSTM(graph.child('lstm'), hidden_size=8, rng=rng)

  x = jnp.ones((2, 5, 4))
  params = rng.seed(bx.Params(), seed=42)

  # Initialize params.
  state, params = lstm.initial_state(params, x[:, 0])
  _, params = lstm.apply(params, x, state)
  params = params.locked()

  # Define the function WITHOUT @jax.jit decorator.
  def forward(params, x, state):
    return lstm.apply(params, x, state)

  # Version 1: is_static=True (uses Python loop internally).
  lstm.is_static = True
  forward_static = jax.jit(forward)
  (out_static, _), _ = forward_static(params, x, state)

  # Modify module for dynamic version.
  lstm.is_static = False

  # Version 2: is_static=False (uses jax.lax.scan internally).
  # This creates a SEPARATE JIT cache.
  forward_dynamic = jax.jit(forward)
  (out_dynamic, _), _ = forward_dynamic(params, x, state)

  # Both outputs have the same shape (semantically equivalent).
  assert out_static.shape == out_dynamic.shape == (2, 5, 8)

  # The key insight: both caches are preserved!
  # Calling forward_static again uses its cached version (is_static=True).
  (out_static_again, _), _ = forward_static(params, x, state)
  assert jnp.allclose(out_static, out_static_again)

  # Calling forward_dynamic uses its cached version (is_static=False).
  (out_dynamic_again, _), _ = forward_dynamic(params, x, state)
  assert jnp.allclose(out_dynamic, out_dynamic_again)

  # Even if we change the module again, neither cache is invalidated.
  lstm.is_static = True  # Change back.

  # forward_static still uses its original cache.
  (out_static_v3, _), _ = forward_static(params, x, state)
  assert jnp.allclose(out_static, out_static_v3)

  # forward_dynamic still uses its original cache (is_static=False was captured).
  (out_dynamic_v3, _), _ = forward_dynamic(params, x, state)
  assert jnp.allclose(out_dynamic, out_dynamic_v3)
