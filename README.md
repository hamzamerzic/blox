<div align="center">
  <img src="https://raw.githubusercontent.com/hamzamerzic/blox/main/images/logo.png" width="400" alt="blox logo">

  <h1>blox</h1>

  <p>
    <strong>A functional and lightweight neural network library for JAX.</strong>
  </p>

  <a href="https://github.com/hamzamerzic/blox/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="blox is released under the MIT license"></a>
  <a href="https://pypi.org/project/jax-blox/"><img src="https://img.shields.io/pypi/v/jax-blox.svg" alt="PyPI"></a>
  <a href="https://blox.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/blox" alt="Documentation Status"></a>
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/jax-0.10+-green" alt="JAX 0.10+">

  <p><strong>📚 Documentation:</strong> <a href="https://blox.readthedocs.io/en/latest/">blox.readthedocs.io</a></p>
</div>

---

**blox** unlocks the full potential of JAX by embracing its functional nature instead of fighting it.

JAX gives you composable transformations over pure functions: write the math, then use `jax.jit`, `jax.grad`, `jax.vmap`, or `jax.checkpoint` on it directly. **blox** is a thin layer on top that adds just enough structure to organize your neural networks, without hiding state behind module objects, global contexts, or framework-specific transform wrappers.

Most JAX libraries try to make JAX feel like PyTorch, forcing an Object-Oriented model on top of it. That is comfortable at first, but it fights JAX's functional nature: it introduces implicit global state and hidden contexts that steepen the learning curve and add cognitive load as your models grow.

**blox** takes the opposite approach and embraces the functional nature of JAX. The entire mental model fits in one line:

```python
outputs, params = model(params, inputs)
```

Parameters go in; outputs and updated parameters come out. This is the standard pattern for [stateful computations in JAX](https://docs.jax.dev/en/latest/stateful-computations.html). Because state flows explicitly through your code, all JAX transformations (`jax.jit`, `jax.grad`, `jax.vmap`, `jax.checkpoint`) work out of the box. No wrappers, no decorators, no surprises.

## 🎯 Who is blox for?

* **Learners:** There is no "framework magic" to learn here. What you see is what you get. It is the best way to understand how neural networks actually work at the JAX level.
* **Practitioners:** If you're tired of fighting frameworks that hide important details, **blox** gives you complete transparency. Whether you're building custom training loops, implementing novel architectures, or scaling up, you have direct access to the full execution stack.

## 📦 Installation

Since blox uses JAX, check the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for your specific hardware.

You will need Python 3.11 or later.

```bash
pip install jax-blox
```

## 🚀 The Basics

### Your First Layer

Let's build a linear layer to see how it feels. Notice the signature: `params` carries all model state, while `inputs` is your data.

```python
import jax
import jax.numpy as jnp
import blox as bx


class Linear(bx.Module):

  def __init__(self, graph: bx.Graph, output_size: int, rng: bx.Rng):
    super().__init__(graph)
    self.output_size = output_size
    self.rng = rng

  def __call__(self, params: bx.Params, x: jax.Array):
    # Parameters are created lazily on first use.
    # No need to specify input shapes upfront or preallocate memory!
    kernel, params = self.get_param(
        params,
        name='kernel',
        shape=(x.shape[-1], self.output_size),
        init=jax.nn.initializers.normal(),
        rng=self.rng,
    )
    bias, params = self.get_param(
        params,
        name='bias',
        shape=(self.output_size,),
        init=jax.nn.initializers.zeros,
        rng=self.rng,
    )
    return x @ kernel + bias, params
```

### Composing Layers

Modules are just Python objects. You can nest them, inject them, or generate them dynamically.

```python
class MLP(bx.Module):

  def __init__(
      self,
      graph: bx.Graph,
      hidden_size: int,
      output_size: int,
      rng: bx.Rng,
  ):
    super().__init__(graph)
    # graph.child('name') creates a unique path for each parameter.
    self.hidden = Linear(graph.child('hidden'), hidden_size, rng=rng)
    self.output = Linear(graph.child('output'), output_size, rng=rng)

  def __call__(self, params: bx.Params, x: jax.Array):
    x, params = self.hidden(params, x)
    x = jax.nn.relu(x)
    return self.output(params, x)
```

### Initialization & Inspection

We cleanly separate "Initialization" (traversing the graph to create parameters) from "Runtime".

```python
# Define the structure.
graph = bx.Graph('net')
rng = bx.Rng(graph.child('rng'))
model = MLP(graph.child('mlp'), hidden_size=128, output_size=10, rng=rng)

# Initialize the parameter container and initialize the RNG state (seed).
# We initialize the RNG first since we use it to initialize other modules.
params = bx.Params()
params = rng.seed(params, seed=42)

# Run a forward pass to trigger lazy parameter initialization.
dummy_input = jnp.ones((1, 784))
_, params = model(params, dummy_input)

# Lock it down to prevent accidental parameter creation during training.
params = params.locked()

# Visualize the full graph and parameter structure.
bx.display(graph, params)
```

## ⚡ JIT Compilation

**blox** modules are trivially compatible with `jax.jit`.

```python
# Just wrap and call. No special decorators needed.
outputs, params = jax.jit(model)(params, inputs)
```

## 📦 The Params Container

The `Params` container holds **all** model state in one place: weights, RNG state, batch norm statistics, moving averages, everything. This is intentional.

**Why put RNG in Params?** In pure functional programming, randomness is state. If your dropout layer consumes a random key, that's a state change. By threading RNG through `Params`, the signature `outputs, params = model(params, inputs)` tells the whole truth: this function might update some internal state.

This design means every parameter is either:

| Type | Examples | Gradients? | Updated during forward? |
|------|----------|------------|------------------------|
| **Trainable** | weights, embeddings | ✅ Yes | No |
| **Non-trainable** | RNG counters, batch norm stats, EMA | ❌ No | Yes |

The `params.split()` method separates these two categories, which becomes important during training.

## 🎯 Training

During training, we want gradients for trainable parameters but also need to capture updates to non-trainable state (like RNG). The pattern is split, run, update, merge:

```python
@jax.jit(donate_argnames='params')
def train_step(params, inputs, targets):
  # Split into trainable (weights) and non-trainable (RNG, etc).
  trainable, non_trainable = params.split()

  def loss_fn(t, nt):
    # Merge to run the forward pass.
    preds, new_params = model(t.merge(nt), inputs)
    loss = jnp.mean((preds - targets) ** 2)

    # Extract non-trainable parameters updated during the forward pass.
    _, new_nt = new_params.split()
    return loss, new_nt

  # Gradients for trainable, updated state for non-trainable.
  grads, new_non_trainable = jax.grad(loss_fn, has_aux=True)(
      trainable, non_trainable
  )

  # Update the trainable parameters using SGD.
  new_trainable = jax.tree.map(lambda w, g: w - 0.01 * g, trainable, grads)

  # Merge updated trainable and updated non-trainable parameters.
  return new_trainable.merge(new_non_trainable)
```

## 🔀 Batching & Parallel RNG

> **⚠️ JAX Sharp Edge**: This section describes patterns needed due to JAX's PRNG design, not blox design decisions. The main sharp edge is around **initialization with `shard_map`** where different parameters may need different sharding. For `vmap` the patterns are straightforward once understood.

Here is a sharp edge in JAX: if you `vmap` or `shard_map` a function that uses random numbers, every batch element/device gets the *same* random key by default. This means your dropout masks would be identical across the whole batch, defeating the purpose of dropout entirely.

**blox** does not hide this behavior from you. Instead, we give you the tools to handle it explicitly: you fold in the lane index when you need unique randomness.

### Understanding JAX's Counter-Based PRNG

JAX's PRNG is stateless and deterministic. When you call `rng(params)`, the returned key is computed as:

```python
new_key = jax.random.fold_in(seed, counter)
```

The `seed` is fixed at initialization, and the `counter` increments with each call. This means:

- **Same seed + same counter = same key** (always)
- **Different counter = different key** (even with same seed)

In parallel contexts (`vmap`, `shard_map`), all lanes share the same seed and counter, so they all get identical keys. To get unique randomness per lane, you must "fold in" the lane index.

### The Manual Folding Pattern

The simplest way to understand RNG folding is to pass the batch index explicitly:

```python
graph = bx.Graph('root')
rng = bx.Rng(graph.child('rng'))
dropout = bx.Dropout(graph.child('dropout'), rate=0.5, rng=rng)

def apply_with_explicit_index(params, x, batch_idx):
  # Fold in the batch index to get a unique seed for this lane.
  original_seed = rng.get_seed(params)
  folded_seed = jax.random.fold_in(original_seed, batch_idx)
  params = rng.seed(params, seed=folded_seed)

  out, params = dropout(params, x, is_training=True)

  # Restore original seed (required for replicated params).
  params = rng.seed(params, seed=original_seed)
  return out, params

# Pass jnp.arange(batch_size) as the index.
batch_indices = jnp.arange(4)
outputs, _ = jax.vmap(
    apply_with_explicit_index,
    in_axes=(None, 0, 0),
    out_axes=(0, None),
)(params, batch_inputs, batch_indices)
```

When using `axis_name` with vmap, you can use `jax.lax.axis_index` instead of threading the index through your code. **This is the recommended approach** as it's more idiomatic:

```python
def apply_with_axis_index(params, x):
  original_seed = rng.get_seed(params)
  folded_seed = jax.random.fold_in(
      original_seed, jax.lax.axis_index('batch')
  )
  params = rng.seed(params, seed=folded_seed)

  out, params = dropout(params, x, is_training=True)

  params = rng.seed(params, seed=original_seed)
  return out, params

# axis_name is required for jax.lax.axis_index.
outputs, _ = jax.vmap(
    apply_with_axis_index,
    in_axes=(None, 0),
    out_axes=(0, None),
    axis_name='batch'
)(params, batch_inputs)
```

### Why Restore the Original Seed?

When `params` is replicated across lanes (`out_axes=None`), JAX requires all lanes to return identical pytrees. If each lane has a different folded seed, JAX will error.

Since we're running the same function in each lane, the counter increments by the same amount everywhere. The seed is the only thing that differs (due to folding), so restoring the original seed ensures the params are identical across all lanes.

### Init vs Runtime

During **initialization**, you typically want identical params across all batch elements, so you do NOT fold in the axis index.

During **runtime**, you want unique randomness per batch element (for dropout, etc.), so you DO fold in the axis index.

You can use `params.is_locked` to detect which mode you're in:

```python
def forward(params, x):
  original_seed = rng.get_seed(params)

  # Check if we're in init mode (unlocked) or runtime mode (locked).
  if params.is_locked:
    # Runtime: fold in axis index for unique randomness.
    original_seed = rng.get_seed(params)
    folded_seed = jax.random.fold_in(
        original_seed, jax.lax.axis_index('batch')
    )
    params = rng.seed(params, seed=folded_seed)

  out, params = dropout(params, x, is_training=True)

  # Restore original seed (no-op during init, required for runtime).
  params = rng.seed(params, seed=original_seed)

  return out, params

# Init phase: params are unlocked, no folding.
def init(x):
  params = rng.seed(bx.Params(), seed=42)
  _, params = forward(params, x)
  return params.locked()

# Both init and runtime use the same vmap.
params = jax.vmap(init, axis_name='batch', out_axes=None)(batch_inputs)

# Runtime phase: params are locked, folding is applied.
outputs, _ = jax.vmap(
    forward,
    in_axes=(None, 0),
    axis_name='batch'
)(params, batch_inputs)
```

This pattern lets you use the same `forward` function for both initialization and runtime.

<details>
<summary><strong>Why use JIT instead of shard_map for initialization?</strong></summary>

When initializing models with sharded parameters, **use `jax.jit` with `out_shardings` rather than `shard_map`**.

`shard_map` is tricky for initialization because:
- Different parameters may need different axes folded in (e.g., model axis but not batch axis)
- Multiple model axes mean different params have different sharding requirements
- Managing which axes to fold for which params becomes complex

`jax.jit` is better because:
- Just specify `out_shardings` and JIT handles partitioning automatically
- Use replicated RNG params during init
- JIT is smart about parameter placement during initialization

```python
# RECOMMENDED: Initialize via JIT with out_shardings
@jax.jit(out_shardings=params_sharding)
def init():
    params = rng.seed(bx.Params(), seed=42)
    _, params = model(params, dummy_input)
    return params.locked()

params = init()  # JIT handles sharding automatically
```

</details>

## 📈 Scaling Up

For models that don't fit on one device, you usually need to shard parameters. **blox** lets you specify sharding as metadata when defining layers, which means you can initialize parameters directly on the correct devices instead of instantiating the full model on the CPU.

```python
from jax.sharding import NamedSharding, PartitionSpec as P

graph = bx.Graph('net')
rng = bx.Rng(graph.child('rng'))

# Define a layer with sharding metadata.
linear = bx.Linear(
    graph.child('linear'),
    output_size=4096,
    rng=rng,
    # Shard weights across the 'model' axis.
    kernel_metadata={'sharding': (None, 'model')},
    bias_metadata={'sharding': ('model',)},
)

def init(x):
  params = rng.seed(bx.Params(), seed=42)
  _, params = linear(params, x)
  return params.locked()

# Get parameter structure without allocating memory or wasting FLOPs.
dummy_input = jnp.ones((4, 4))
abstract_params = jax.eval_shape(init, dummy_input)

# Build sharding specs (assuming 2*2=4 GPU/TPU devices).
mesh = jax.make_mesh((2, 2), ('batch', 'model'))

params_sharding = jax.tree.map(
    lambda p: NamedSharding(mesh, P(*p.sharding)),
    abstract_params,
    is_leaf=lambda x: isinstance(x, bx.Param),
)

 # Example data and output sharding
data_sharding = NamedSharding(mesh, P('batch'))
output_sharding = NamedSharding(mesh, P('batch'))

# Initialize directly on device using out_shardings.
sharded_init = jax.jit(init, out_shardings=params_sharding)
sharded_params = sharded_init(dummy_input)

# Define forward pass with full input/output sharding.
@jax.jit(
    in_shardings=(params_sharding, data_sharding),
    out_shardings=(output_sharding, params_sharding)
)
def forward(params, x):
  return linear(params, x)

out, new_params = forward(sharded_params, dummy_input)
```

## 🔄 Recurrence & Scanning

**blox** provides two base classes for sequence processing:

* **`SequenceBase`**: For models like Transformers where you process the whole sequence at once.
* **`RecurrenceBase`**: For models like LSTMs where you iterate step-by-step.

The built-in `LSTM` and `GRU` extend `RecurrenceBase`. They are designed to work with `jax.lax.scan` for maximum efficiency:

```python
lstm = bx.LSTM(graph.child('lstm'), hidden_size=256, rng=rng)

# Initialize the LSTM state.
inputs = jnp.ones((batch_size, seq_len, features))
state, params = lstm.initial_state(params, inputs[:, 0])

# Process the whole sequence efficiently.
(outputs, final_state), params = lstm.apply(
    params, inputs, prev_state=state
)
```

## 🧠 Under the Hood

**blox** is transparent by design. The abstraction is just automated path handling to keep your code clean and your state pure.

* **Graph**: Defines the model hierarchy (e.g., `net -> mlp -> hidden`). `graph.child('name')` creates child nodes, giving each module a unique path for namespacing its parameters. The graph stores references to all created modules and provides `graph.walk()` for iteration, which is useful for applying LoRA adapters or toggling training mode across layers.

* **Module**: Has a unique path in the graph and provides convenience methods (`get_param`, `set_param`) to automatically manage its own parameters.

* **Param**: A wrapper around each parameter value that holds a `trainable` flag and arbitrary `metadata`. The trainable flag determines whether a parameter is differentiable or not.

* **Params**: Immutable container holding all state as a flat dictionary of `Param` objects keyed by tuple paths (e.g., `('net', 'mlp', 'hidden', 'kernel')`). Use `split()` to separate trainable from non-trainable state.

* **Rng**: A module that generates deterministic random keys. Since it's used to randomly initialize all other parameters and provide runtime randomness, it must be seeded first via `rng.seed(params, seed=42)`.

## ⚖️ Why the verbosity?

**blox chooses clarity over brevity.**

Most frameworks rely on implicit global state or thread-local contexts to hide parameters. That saves a few keystrokes, but it creates a "black box."

| OOP-style Wrappers | **blox** |
| --- | --- |
| `out = layer(x)` | `outputs, params = layer(params, inputs)` |
| Implicit global state | Explicit state passing |
| Opaque variable scopes | Explicit `bx.Graph` paths |
| Custom `vmap` / `jit` / ... wrappers | Standard `jax.vmap` / `jax.jit` / ... |

By accepting slightly more verbose function signatures, you gain:

1. **Total transparency:** You know exactly what data your function touches.
2. **Full control:** No global state means no unknown side-effects.
3. **Maximum performance:** Zero overhead.

## 🔀 Decoupled Params and Graph

A key design principle is the **clean separation between parameters and the model graph**. Unlike other libraries where params are tightly coupled to modules, blox lets multiple models share the same params. That separation buys you a few things:

1. **No single module owns params.** Params are passed in, not stored. Multiple models can use the same params without ownership conflicts.
2. **Avoids pytree complexity.** Modules as pytrees containing both static config and JAX arrays require magic handling, especially for non-hashable types (like lists and dicts). blox keeps a clean split: `Graph` is static (Python objects), `Params` is dynamic (JAX arrays).
3. **Graph is static, params are dynamic.** Graph describes *what* operations to do. Params provide *what values* to use. This separation is maintained throughout execution.

**Use cases:**

* **Actor vs Learner in RL**: Separate models for data collection and training that share weights.
* **Training vs Evaluation**: Scenarios where evaluation logic differs significantly while relying on the same parameters.

**Rule of thumb:** If the model needs recompilation, prefer creating a new model instead of modifying an existing one or creating an Uber-module. Let the function shape the design: if you need both static and dynamic scanning behavior, create two models.

```python
# Create two LSTM variants with the same parameter structure.
def create_lstm(is_static: bool):
  graph = bx.Graph('model')
  rng = bx.Rng(graph.child('rng'))
  return bx.LSTM(
      graph.child('lstm'),
      hidden_size=64,
      rng=rng,
      is_static=is_static
  )

lstm_static = create_lstm(is_static=True)  # Python loop (debuggable).
lstm_dynamic = create_lstm(is_static=False)  # lax.scan (production).

# Both models work with the SAME params!
out_static, _ = lstm_static.apply(params, inputs, prev_state=state)
out_dynamic, _ = lstm_dynamic.apply(params, inputs, prev_state=state)
```

While nothing prevents users from changing modules in place, JAX will not recompile functions automatically unless manually instructed (see [JAX Gotchas](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#using-jax-jit-with-class-methods)). We do support exceptions for ergonomics, such as the `is_training` flag when evaluation logic is only a simple flag away from training logic, e.g. dropout.

## 🔬 How blox compares: Equinox & Flax NNX

JAX already ships a strong abstraction: composable transformations over pure functions, with [state threaded explicitly through function signatures](https://docs.jax.dev/en/latest/stateful-computations.html). The question a neural-network library answers is *how much of that stays visible*. Two well-built libraries take different routes, and both make simple things easy while adding overhead as models get complicated.

**[Equinox](https://docs.kidger.site/equinox/)** keeps the model *as* a PyTree, which is close to JAX in spirit. But because the module PyTree mixes array leaves with arbitrary Python objects, plain `jax.jit` / `jax.grad` don't apply directly: ["it only makes sense to trace arrays."](https://docs.kidger.site/equinox/faq/) You switch to a parallel family of *filtered* transforms (`filter_jit`, `filter_grad`, `filter_vmap`, …), learn `partition` / `combine`, and learn which default filter (`is_array` vs `is_inexact_array`) decides what gets gradients. Stateful layers like `BatchNorm` are threaded by hand, and shared/tied layers are a value-semantics foot-gun.

**[Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html)** goes the other way: mutable, PyTorch-style module objects with reference semantics. Because ["JAX transformations operate on pytrees of `jax.Array`s and abide by value semantics,"](https://flax.readthedocs.io/en/latest/guides/transforms.html) NNX maintains its *own* transforms (`nnx.jit`, `nnx.grad`, `nnx.vmap`, `nnx.scan`, `nnx.remat`, …, ["supersets of their equivalent JAX counterparts"](https://flax.readthedocs.io/en/latest/nnx_basics.html)), plus a `Module` / `State` / `GraphDef` system and a `split` / `merge` ceremony for "crossing boundaries." That is a real maintenance surface and a leaky abstraction: mutation ["must be used with care because it can clash with JAX's underlying assumptions,"](https://flax.readthedocs.io/en/latest/guides/transforms.html) and the maintainers' own plan is to eventually make NNX ["implement the pytree protocol"](https://github.com/google/flax/discussions/4736) so it can be used with raw JAX transforms.

**blox** avoids both boundaries by construction. `Graph` (static Python) and `Params` (dynamic arrays) are *separate objects*, so there is no mixed-leaf tree to filter and no mutable graph to split and merge. `jax.jit`, `jax.grad`, `jax.vmap`, and `jax.checkpoint` apply to a plain function: no `filter_*`, no `nnx.*` wrapper in the way.

Randomness is the one sharp edge every JAX library inherits, and the three handle it differently. Equinox threads `jax.random` keys through your functions by hand. NNX hides them inside a stateful `nnx.Rngs` object whose keys live in the graph and need ["extra tricks with `nnx.vmap`"](https://flax.readthedocs.io/en/latest/guides/randomness.html) to behave correctly under transforms. blox keeps JAX's own counter-based `fold_in` pattern and [surfaces it explicitly](#-batching--parallel-rng) rather than wrapping it: the sharp edge is JAX's, and so is everything you learn working around it.

| | Equinox | Flax NNX | **blox** |
|---|---|---|---|
| Calls JAX transforms directly | Mostly: non-array leaves force `filter_*` | No: `nnx.jit` / `nnx.scan` / … reimplementations | **Yes: `jax.jit`/`grad`/`vmap`/`checkpoint`, unwrapped** |
| Boundary ceremony | `partition` / `combine`, filter specs | `nnx.split` / `nnx.merge` (State / GraphDef) | **None: `params` is already a clean array pytree** |
| Where state lives | In the module PyTree | In mutable `Module` instances (a graph) | **In a separate `Params` container** |
| Library-specific transforms to maintain | filtered transforms | a full parallel transform suite | **Zero** |
| Randomness | manual `jax.random` key threading | stateful `nnx.Rngs` in the graph (`split_rngs` / `StateAxes` for `vmap`/`scan`) | **JAX's own `fold_in` pattern, surfaced** |

Both Equinox and NNX are mature and a great fit for many projects: Equinox if you like "the model is a PyTree," NNX if mutable PyTorch-style objects feel natural. **blox makes a different bet: rather than building a framework on top of JAX, it grows directly out of JAX's own philosophy of explicit state, pure functions, and no hidden magic.** The graph and the parameters stay separate, every transformation is the real `jax.*` one, and the randomness is JAX's own. What you learn using blox is JAX itself, so your understanding and your code keep paying off as the ecosystem moves, with nothing library-specific standing in the way.

## 📄 License

MIT. See [LICENSE](LICENSE).
