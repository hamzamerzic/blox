<div align="center">
  <img src="https://i.ibb.co/FLmR2T3r/logo.png" width="400" alt="blox logo">

  <h1>blox</h1>

  <p>
    <strong>A functional and lightweight neural network library for JAX.</strong>
  </p>

  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="blox is released under the MIT license"></a>
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/jax-0.8+-green" alt="JAX 0.8+">
</div>

---

**blox** unlocks the full potential of JAX by embracing its functional nature instead of fighting it.

JAX gives you composable transformations that let you write math and have it run blazingly fast on any hardware. **blox** is a thin layer on top that keeps all of that power accessible while giving you just enough structure to organize your neural networks.

Most JAX libraries try to force Object-Oriented paradigms to make JAX feel like PyTorch. While comfortable at first, this fights against JAX's functional nature. It introduces implicit global state and hidden contexts that eventually increase cognitive load and steepen the learning curve.

**blox** takes the opposite approach. We embrace the functional nature of JAX.

The entire mental model fits in one line:

```python
outputs, params = model(params, inputs)
```

Parameters go in, outputs and updated parameters come out. This is the standard pattern for [stateful computations in JAX](https://docs.jax.dev/en/latest/stateful-computations.html). Because state flows explicitly through your code, all JAX transformations—`jax.jit`, `jax.grad`, `jax.vmap`, `jax.checkpoint`—work out of the box. No wrappers, no decorators, no surprises.

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
        init=jax.nn.initializers.glorot_uniform(),
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
    # graph.child('name') creates a unique path for parameters
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
# We need the RNG to initialize parameters so we initialize it first.
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

The `Params` container holds **all** model state in one place: weights, RNG state, batch norm statistics, moving averages—everything. This is intentional.

**Why put RNG in Params?** In pure functional programming, randomness is state. If your dropout layer consumes a random key, that's a state change. By threading RNG through `Params`, the signature `outputs, params = model(params, inputs)` tells the whole truth: this function might update some internal state.

This design means every parameter is either:

| Type | Examples | Gradients? | Updated during forward? |
|------|----------|------------|------------------------|
| **Trainable** | weights, embeddings | ✅ Yes | No |
| **Non-trainable** | RNG counters, batch norm stats, EMA | ❌ No | Yes |

The `params.split()` method separates these two categories, which becomes important during training.

## 🎯 Training

During training, we want gradients for trainable parameters but also need to capture updates to non-trainable state (like RNG). The pattern:

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

Here is a sharp edge in JAX: if you `vmap` or `shard_map` a function that uses random numbers, every batch element/device gets the *same* random key by default. This means your dropout masks would be identical across the whole batch.

**blox** does not hide this behavior from you. Instead, we give you the tools to handle it explicitly.

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

When using `axis_name` with vmap, you can use `jax.lax.axis_index` instead of threading the index through your code:

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

You can use `params._locked` to detect which mode you're in:

```python
def forward(params, x):
  is_runtime = params._locked

  if is_runtime:
    # Runtime: fold in axis index for unique randomness.
    original_seed = rng.get_seed(params)
    folded_seed = jax.random.fold_in(
        original_seed, jax.lax.axis_index('batch')
    )
    params = rng.seed(params, seed=folded_seed)

  out, params = dropout(params, x, is_training=is_runtime)

  if is_runtime:
    # Restore original seed for replicated params.
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

## 📈 Scaling Up

For models that don't fit on one device, you usually need to shard parameters. **blox** lets you specify sharding as metadata when defining layers.

We can use this to initialize parameters directly on the correct devices, avoiding the need to instantiate the full model on the CPU.

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

* **Graph**: Defines the model hierarchy (e.g., `net -> mlp -> hidden`). `graph.child('name')` creates child nodes, giving each module a unique path for namespacing its parameters. The graph stores references to all created modules and provides `graph.walk()` for iteration—useful for applying LoRA adapters or toggling training mode across layers.

* **Module**: Has a unique path in the graph and provides convenience methods (`get_param`, `set_param`) to automatically manage its own parameters.

* **Param**: A wrapper around each parameter value that holds a `trainable` flag and arbitrary `metadata`. The trainable flag determines whether a parameter is differentiable or not.

* **Params**: Immutable container holding all state as a flat dictionary of `Param` objects keyed by tuple paths (e.g., `('net', 'mlp', 'hidden', 'kernel')`). Use `split()` to separate trainable from non-trainable state.

* **Rng**: A module that generates deterministic random keys. Since it's used to randomly initialize all other parameters and provide runtime randomness, it must be seeded first via `rng.seed(params, seed=42)`.

## ⚖️ Why the verbosity?

**blox chooses clarity over brevity.**

Most frameworks rely on implicit global state or thread-local contexts to hide parameters. While this saves a few keystrokes, it creates a "black box."

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

## 📄 License

MIT. See [LICENSE](LICENSE).