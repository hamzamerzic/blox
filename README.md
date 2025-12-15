<div align="center">
  <img src="https://i.ibb.co/FLmR2T3r/logo.png" width="400" alt="blox logo">
  
  <h1>blox</h1>
  
  <p>
    <strong>A functional and lightweight neural network library for JAX.</strong>
  </p>

  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="blox is released under the MIT license"></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/jax-0.4+-green" alt="JAX 0.4+">
</div>

---

**blox** unlocks the full potential of JAX by embracing its functional nature instead of fighting it.

Most JAX neural network libraries try to force Object-Oriented paradigms to make JAX feel like PyTorch, usually by introducing implicit global state, hidden contexts, or clever magic that seems helpful at first but eventually results in unnecessary cognitive overhead and a steep learning curve.

**blox** takes the opposite approach. Instead of hiding JAX's functional approach, it leans into it, building a minimal abstraction layer on top. By stripping away the "magic", **blox** ensures explicit data flow and keeps your code transparent, free of side effects, and trivially compatible with JAX's powerful transformations.

## ⚡ Core Principles & Features

* **Native JAX Compatibility:** Works with all JAX transformations, including `jax.jit`, `jax.grad`, `jax.vmap`, `jax.shard_map`, `jax.checkpoint`, and others. No special wrappers or decorators are required.
* **Functional Purity:** Models are stateless transformations. Parameters are explicit arguments, never hidden in `self` or global registries.
* **Explicit Data Flow:** Every function returns `(outputs, params)`, making data dependencies crystal clear and eliminating side effects. You can trace the path of every single tensor just by reading the function signature.
* **Lazy Initialization:** Define your model structure abstractly, then run a single forward pass to materialize parameters automatically.
* **Structural RNG Keys:** Randomness is handled as part of the `Params` structure. Getting a new random key simply returns an updated `Params` object, ensuring deterministic reproducibility without the boilerplate of manually threading keys.
* **Interactive Inspection:** Debugging is easier when you can see your model. **blox** integrates with **Treescope** to let you interactively inspect your model's architecture, hierarchy, and parameter shapes.

## 📦 Installation

Since blox uses JAX, check out the [JAX installation instructions](https://jax.readthedocs.io/en/latest/installation.html) for your specific hardware (CPU/GPU/TPU).

You will need Python 3.10 or later. Install blox from PyPi:

```bash
pip install jax-blox
```

## 🚀 Quick Start

In **blox**, a module is just a structural container (`__init__`) and a set of pure mathematical functions (like `__call__`).

### Define your layers

Notice the signature: `params` carries the state (weights + RNG), while `inputs` is your data.

```python
import jax
import jax.numpy as jnp
import blox as bx

class CustomLinear(bx.Module):

  def __init__(
      self,
      graph: bx.Graph,
      output_size: int,
  ) -> None:
    super().__init__(graph)
    self.output_size = output_size

  def __call__(
      self,
      params: bx.Params,
      inputs: jax.Array,
  ) -> tuple[jax.Array, bx.Params]:
    # Param initialization is lazy which serves two important purposes:
    # 1. Avoids the need to specify input dimensions at construction.
    # 2. Prevents accidental allocation of params on device.
    kernel, params = self.get_param(
        params=params,
        name='kernel',
        shape=(inputs.shape[-1], self.output_size),
        init=jax.nn.initializers.glorot_uniform()
    )
    bias, params = self.get_param(
        params=params,
        name='bias',
        shape=(self.output_size,),
        init=jax.nn.initializers.zeros
    )
    return inputs @ kernel + bias, params
```

### Composition & Dependency Injection

Because **blox** modules are standard Python objects, composing them via dependency injection is intuitive.

Instead of hardcoding layers, you can inject them. The injected modules keep their original position in the hierarchy, while internal layers become children.

```python
class CustomMLP(bx.Module):

  def __init__(
      self,
      graph: bx.Graph,
      hidden_size: int,
      # We can inject externally created modules...
      output_projection: bx.Module,
  ) -> None:
    super().__init__(graph)
    # ... or create new ones internally.
    self.hidden_proj = CustomLinear(graph.child('hidden'), hidden_size)
    self.output_projection = output_projection

  def __call__(
      self,
      params: bx.Params,
      inputs: jax.Array,
  ) -> tuple[jax.Array, bx.Params]:
    # Chain the functional transformations.
    hidden, params = self.hidden_proj(params, inputs)
    hidden = jax.nn.relu(hidden)
    return self.output_projection(params, hidden)
```

### Initialization & Inspection

We cleanly separate the "Initialization phase" (traversing the graph to create parameters) from the "Runtime phase" (updating trainable and non-trainable parameters).

```python
# Define the structure for wiring modules.
graph = bx.Graph('net')

# Create the output layer explicitly and use it to create our CustomMLP.
readout = CustomLinear(graph.child('readout'), output_size=1)
model = CustomMLP(graph.child('mlp'), hidden_size=32, output_projection=readout)

# Create dummy input data to infer shapes.
inputs = jnp.ones((1, 10))

# Initialize the parameters.
# Params requires an Rng module for handling randomness.
rng = bx.Rng(graph.child('rng'), seed=42)
params = bx.Params(rng=rng)

# Run a forward pass to trigger lazy initialization.
unused_outputs, params = model(params, inputs)

# Finalize Params to prevent accidental structure changes later.
params = params.finalize()

# Visualize the full graph and parameter structure.
bx.display(graph, params)
```

**Output:**
Notice how `readout` and `mlp` are siblings in the graph, while `hidden` is nested inside `mlp`.

```text
net: Graph # Param: 387 (1.5 KB)(
  readout=CustomLinear # Param: 33 (132 B)(
    output_size=1,
    kernel=Param[T](shape=(32, 1), dtype='float32', value=<jax.Array float32(32, 1) ≈-0.048 ±0.21 [≥-0.38, ≤0.35] nonzero:32>),
    bias=Param[T](shape=(1,), dtype='float32', value=<jax.Array([0.], dtype=float32)>),
  ),
  mlp=CustomMLP # Param: 352 (1.4 KB)(
    hidden_size=32,
    output_projection=Link(path='net/readout'),
    hidden=CustomLinear # Param: 352 (1.4 KB)(output_size=32, kernel=Param[T](shape=(10, 32), dtype='float32', value=<jax.Array float32(10, 32) ≈-0.0016 ±0.22 [≥-0.37, ≤0.38] nonzero:320>), bias=Param[T](shape=(32,), dtype='float32', value=<jax.Array float32(32,) ≈0.0 ±0.0 [≥0.0, ≤0.0] zero:32>)),
  ),
  rng=Rng # Param: 2 (12 B)(
    seed=42,
    key=Param[N](shape=(), dtype='key<fry>', value=<jax.Array key<fry>()>),
    counter=Param[N](shape=(), dtype='uint32', value=<jax.Array(2, dtype=uint32)>),
  ),
)
```

## 🔀 Parallel Execution (vmap & shard\_map)

JAX's `jit` handles RNG splitting automatically. However, when using explicit parallelization like `jax.vmap` or `jax.shard_map`, you want distinct behavior on each device or batch element (e.g. unique dropout masks or params per shard).

If you simply passed the same `params` (and thus the same RNG state) to every device, they would all produce identical random numbers. **blox** solves this by letting you "fold in" axes. This keeps the base RNG state replicated (identical across devices) but mixes in the device index to generate unique keys per device.

```python
def apply_model(params, inputs):
  # Fold in the batch axis so each batch element gets a unique RNG stream.
  params = params.fold_in_axes('batch')
  outputs, params = dropout(params, inputs, is_training=True)
  # Fold out before returning to restore the replicated state structure.
  return outputs, params.fold_out_axes('batch')

# Note that params (including the Rng) are replicated.
batched_outputs = jax.vmap(
    apply_model,
    in_axes=(None, 0),
    out_axes=(0, None),
    axis_name='batch'
)(params, inputs)
```

## 🏷️ Parameter Metadata & Sharding

To initialize large models efficiently, we must create parameters directly on their target devices. **blox** supports this via `jax.eval_shape` and explicit `out_shardings`.

```python
from jax.sharding import NamedSharding, PartitionSpec as P
import functools

graph = bx.Graph('net')
linear = bx.Linear(
  graph.child('linear'),
  output_size=1024,
  kernel_metadata={'sharding': (None, 'model')},
  bias_metadata={'sharding': ('model',)},
)
rng = bx.Rng(graph.child('rng'), 42)

# Define an initialization function.
def init(x):
  _, params = linear(bx.Params(rng=rng), x)
  return params.finalize()

# Abstract evaluation to get the Params structure (no memory allocation).
inputs = jnp.ones((4, 4))
abstract_params = jax.eval_shape(init, inputs)

# Create the sharding specification from metadata.
mesh = jax.make_mesh((4,), ('model',))

params_sharding = jax.tree.map(
    lambda p: NamedSharding(mesh, P(*p.sharding)),
    abstract_params,
    is_leaf=lambda x: isinstance(x, bx.Param)
)

# JIT-compile the init function with out_shardings.
# Params are created directly on the correct devices, with no memory overhead.
sharded_init = jax.jit(init, out_shardings=params_sharding)
sharded_params = sharded_init(inputs)

@functools.partial(jax.jit, in_shardings=(params_sharding, None))
def forward(params, x):
  return linear(params, x)

out, new_params = forward(sharded_params, inputs)
```

## 🔄 Recurrence & Scanning

Managing state in RNNs with JAX usually requires complex `jax.lax.scan` boilerplate. **blox** modules like `bx.LSTM` simplify this by providing both a step-wise `__call__` and a sequence-processing `apply`.

```python
lstm = bx.LSTM(graph.child('lstm'), hidden_size=128)

# Initialize the LSTM state.
state, params = lstm.initial_state(params, inputs)

# Run efficient compiled scan over a sequence [Batch, Time, Features].
# It automatically handles carry propagation.
(outputs, final_state), params = lstm.apply(
    params, inputs_sequence, prev_state=state
)
```

## ⚡ Training (JIT & Gradients)

The `Params` container holds *everything*: weights, RNG state, batch norm statistics, EMA moving averages, ...

When training, we usually want to differentiate w.r.t. trainable parameters, such as weights, but still update non-trainable parameters like the RNG state. **blox** makes this partitioning explicit and simple.

```python
@jax.jit
def train_step(params, inputs, targets):
  # Split params into two sets.
  # Trainable: weights, biases (we want gradients for these).
  # Non-trainable: Rng, batch stats, EMA (we just want the updated values).
  trainable, non_trainable = params.split()

  def loss_fn(t, nt):
    # Merge parameters to run the forward pass.
    predictions, new_params = model(t.merge(nt), inputs)

    # Calculate the loss.
    loss = jnp.mean((predictions - targets) ** 2)

    # Extract the updated non-trainable state to pass it out.
    _, new_non_trainable = new_params.split()
    return loss, new_non_trainable

  # Calculate gradients and capture the auxiliary state (non_trainable updates).
  grads, new_non_trainable = jax.grad(loss_fn, has_aux=True)(
      trainable, non_trainable
  )

  # Update the trainable weights using SGD.
  new_trainable = jax.tree.map(lambda w, g: w - 0.01 * g, trainable, grads)

  # Merge the updated weights with the updated non-trainable state.
  return new_trainable.merge(new_non_trainable)
```

## 🧠 Under the Hood

**blox** is transparent by design. The abstraction is really just automated path handling to keep your code clean and your state pure.

* **The Graph:** A lightweight object representing a location in the hierarchy (e.g. `net -> mlp -> dense1`). `graph.child('name')` appends to the path, ensuring every module has a unique address space.
* **The Params:** A flat, immutable dictionary holding all state, keyed by tuple paths (e.g. `('net', 'mlp', 'dense1', 'kernel')`). It supports simple partitioning for gradients or custom metadata.
* **The Rng:** `Params` maintains an Rng module such that when a module requests randomness, `Params` generates a unique, deterministic key via the Rng module and an *updated* `Params` structure. Modules, such as `Dropout` can use a custom Rng module, to decouple their randomness from the `Params` randomness.

## ⚖️ Why blox?

**blox chooses clarity over brevity.**

Most frameworks rely on implicit global state or thread-local contexts to hide parameters and RNG keys. While this makes simple scripts shorter, it creates a "black box" that is hard to debug and even harder to customize.

| OOP-style Wrappers | **blox** |
| :--- | :--- |
| `out = layer(x)` | `outputs, params = layer(params, inputs)` |
| Implicit global state | Explicit state passing |
| Opaque variable scopes | Explicit `bx.Graph` paths |
| Custom `vmap` / `jit` / `grad` wrappers | Standard `jax.vmap` / `jax.jit` / `jax.grad` |

By accepting slightly more verbose function signatures, you gain:

1.  **Total Transparency:** You know exactly what data your function touches.
2.  **JIT Safety:** No global state means no side-effect leaks or tracer errors.
3.  **Maximum Performance:** Zero overhead abstractions.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
