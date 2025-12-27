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

Most JAX neural network libraries try to force Object-Oriented paradigms to make JAX feel like PyTorch, usually by introducing implicit global state, hidden contexts, or clever magic that seems helpful at first but eventually results in unnecessary cognitive overhead and a steep learning curve.

**blox** takes the opposite approach. Instead of hiding JAX's functional approach, it leans into it, building a minimal abstraction layer on top. By stripping away the "magic", **blox** ensures explicit data flow and keeps your code transparent, free of side effects, and trivially compatible with JAX's powerful transformations.

## 🎯 Who is blox for?

**blox** is mainly designed for:

* **Students** who want to learn JAX without having to learn a ton of additional abstractions. With **blox**, what you see is what you get: pure functions, explicit state, and no hidden magic. This makes it an excellent educational tool for understanding how neural networks actually work at the JAX level.

* **Practitioners** who want full control over the execution stack. If you're tired of fighting frameworks that hide important details, **blox** gives you complete transparency while still providing the conveniences you need for building high performance models.

## ⚡ Core Principles & Features

* **Native JAX compatibility:** Works with all JAX transformations, including `jax.jit`, `jax.grad`, `jax.vmap`, `jax.shard_map`, `jax.checkpoint`, and others. No special wrappers or decorators are required.
* **Functional purity:** Models are stateless transformations. Parameters are explicit arguments, never hidden in `self` or global registries.
* **Explicit data flow:** Every function returns `(outputs, params)`, making data dependencies crystal clear and eliminating side effects. You can trace the path of every single tensor just by reading the function signature.
* **Lazy initialization:** Define your model structure abstractly, then run a single forward pass to materialize parameters automatically.
* **Structural RNG keys:** Randomness is handled as part of the `Params` structure. Getting a new random key simply returns an updated `Params` object, ensuring deterministic reproducibility without the boilerplate of manually threading keys.
* **Interactive inspection:** Debugging is easier when you can see your model. **blox** integrates with **Treescope** to let you interactively inspect your model's architecture, hierarchy, and parameter shapes.

## 📦 Installation

Since blox uses JAX, check out the [JAX installation instructions](https://jax.readthedocs.io/en/latest/installation.html) for your specific hardware (CPU/GPU/TPU).

You will need Python 3.11 or later. Install blox from PyPi:

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
      rng: bx.Rng,
  ) -> None:
    super().__init__(graph)
    self.output_size = output_size
    self.rng = rng

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
        init=jax.nn.initializers.glorot_uniform(),
        rng=self.rng,
    )
    bias, params = self.get_param(
        params=params,
        name='bias',
        shape=(self.output_size,),
        init=jax.nn.initializers.zeros,
        rng=self.rng,
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
      rng: bx.Rng,
  ) -> None:
    super().__init__(graph)
    # ... or create new ones internally.
    self.hidden_proj = CustomLinear(graph.child('hidden'), hidden_size, rng=rng)
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
rng = bx.Rng(graph.child('rng'))

# Create the output layer explicitly and use it to create our CustomMLP.
output_projection = CustomLinear(graph.child('linear'), output_size=1, rng=rng)
model = CustomMLP(
    graph.child('mlp'),
    hidden_size=32,
    output_projection=output_projection,
    rng=rng,
)

# Create dummy input data to infer shapes.
inputs = jnp.ones((1, 10))

# Initialize the parameters.
params = bx.Params()  # Create empty container to hold the full model state.
params = rng.seed(params, seed=42)  # Initialize the Rng's state.

# Run a forward pass to trigger lazy initialization to populates Params.
unused_outputs, params = model(params, inputs)

# Finalize to prevent accidentally adding new parameters in the future and to
# be able to use params.initialized property to control the execution flow.
params = params.finalized()

# Visualize the full graph and parameter structure.
bx.display(graph, params)
```

**Output:**
Notice how `linear` and `mlp` are siblings in the graph, while `hidden` is nested inside `mlp`.
The `output_projection` in `mlp.__init__` shows a reference to `linear`'s constructor.

<details open>
<summary><b>net: Graph</b> <i># Param: 387 (1.5 KB)</i></summary>

<blockquote>
<details open>
<summary><b>linear</b>=<b>CustomLinear</b> <i># Param: 33 (132 B)</i></summary>
<pre>
__init__=CustomLinear(output_size=1)
kernel=Param[T](shape=(32, 1), dtype=float32, value=≈-0.048 ±0.21)
bias=Param[T](shape=(1,), dtype=float32, value=0.0)
</pre>
</details>

<details open>
<summary><b>mlp</b>=<b>CustomMLP</b> <i># Param: 352 (1.4 KB)</i></summary>
<pre>
__init__=CustomMLP(hidden_size=32, output_projection=<a href="#linear">CustomLinear(output_size=1)</a>)
</pre>
<blockquote>
<details>
<summary><b>hidden</b>=<b>CustomLinear</b> <i># Param: 352 (1.4 KB)</i></summary>
<pre>
__init__=CustomLinear(output_size=32)
kernel=Param[T](shape=(10, 32), dtype=float32, value=≈-0.0016 ±0.22)
bias=Param[T](shape=(32,), dtype=float32, value=0.0)
</pre>
</details>
</blockquote>
</details>

<details>
<summary><b>rng</b>=<b>Rng</b> <i># Param: 2 (12 B)</i></summary>
<pre>
__init__=Rng()
seed=Param[N](shape=(), dtype=key, metadata={'tag': 'rng_seed'})
counter=Param[N](shape=(), dtype=uint32, metadata={'tag': 'rng_counter'}, value=2)
</pre>
</details>
</blockquote>

</details>

## ⚡ JIT Compilation

Since **blox** modules are pure functions with no hidden state, they work directly with `jax.jit`:

```python
# Just wrap and call - no special decorators needed.
outputs, params = jax.jit(model)(params, inputs)
```

## 🎯 Training

The `Params` container holds *everything*: weights, RNG state, batch norm statistics, EMA moving averages, ...

When training, we usually want to differentiate w.r.t. trainable parameters, such as weights, but still update non-trainable parameters like the RNG state. **blox** makes this partitioning explicit and simple.

```python
@jax.jit(donate_argnames='params')
def train_step(params, inputs, targets):
  # Split params into two sets:
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

## 🔀 Batching & Parallel RNG (vmap & shard\_map)

JAX's `jit` handles RNG splitting automatically. However, when using explicit parallelization like `jax.vmap` or `jax.shard_map`, you want distinct behavior on each device or batch element (e.g. unique dropout masks or params per shard).

If you simply passed the same `params` (and thus the same RNG state) to every device, they would all produce identical random numbers. **blox** solves this by automatically "folding in" axes when using `bx.Rng(..., auto_fold_in_axes=True)`. This keeps the base RNG state replicated (identical across devices) but folds in the device/batch index to generate unique keys per device.

```python
# By default, Rng automatically folds in vmap/shard_map axes.
rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=True)

def apply_model(params, inputs):
  # No manual folding needed! Rng detects the 'batch' axis from vmap.
  outputs, params = dropout(params, inputs, is_training=True)
  return outputs, params

# Note that params (including the Rng) are replicated.
batched_outputs = jax.vmap(
    apply_model,
    in_axes=(None, 0),
    out_axes=(0, None),
    axis_name='batch'
)(params, inputs)
```

### Manual RNG Control

For full control over how RNG keys are generated (e.g. specialized folding or
key manipulation), you can disable automatic folding.

```python
# Disable automatic folding for manual control.
rng = bx.Rng(graph.child('rng'), auto_fold_in_axes=False)

def apply_model(params, inputs):
  # Manually get and manipulate the seed.
  original_seed = rng.get_seed(params)
  folded_seed = jax.random.fold_in(original_seed, jax.lax.axis_index('batch'))
  
  # Update params with the new seed.
  params = rng.seed(params, seed=folded_seed)
  
  # Forward pass uses the manual seed.
  outputs, params = model(params, inputs)
  
  # Restore the original seed before returning to keep state replicated.
  return outputs, rng.seed(params, seed=original_seed)
```

## 📈 Scaling Up

To run models that don't fit on a single device, parameters must be created directly on their target devices. This requires specifying how parameters are sharded. You can do this manually by inspecting the parameter structure, or automatically by baking sharding metadata into the model definition.

Below we show how to specify sharding metadata per module and extract it without any FLOPs using `jax.eval_shape`.

```python
from jax.sharding import NamedSharding, PartitionSpec as P

graph = bx.Graph('net')
rng = bx.Rng(graph.child('rng'))
linear = bx.Linear(
  graph.child('linear'),
  output_size=1024,
  rng=rng,
  kernel_metadata={'sharding': (None, 'model')},
  bias_metadata={'sharding': ('model',)},
)

# Define an initialization function.
def init(x):
  params = rng.seed(bx.Params(), seed=42)
  _, params = linear(params, x)
  return params.finalized()

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

@jax.jit(in_shardings=(params_sharding, None), donate_argnames='params')
def forward(params, x):
  return linear(params, x)

out, new_params = forward(sharded_params, inputs)
```

## 🔄 Recurrence & Scanning

**blox** provides `bx.SequenceBase` for general sequence models that handle sequence data using a step-wise `__call__` and a sequence-processing `apply`.
`bx.RecurrenceBase` is a subclass of `bx.SequenceBase` where sequence-processing `apply` function by default iteratively applies `__call__`.
In simple terms, `bx.SequenceBase` should be used to implement a Transformer model and `bx.RecurrenceBase` to implement a standard `LSTM`.

```python
lstm = bx.LSTM(graph.child('lstm'), hidden_size=128, rng=rng)

# Initialize the LSTM state.
state, params = lstm.initial_state(params, inputs)

# Run efficient compiled scan over a sequence [Batch, Time, Features].
# It automatically handles carry propagation.
(outputs, final_state), params = lstm.apply(
    params, inputs_sequence, prev_state=state
)
```

## 🧠 Under the Hood

**blox** is transparent by design. The abstraction is really just automated path handling to keep your code clean and your state pure.

* **The Graph:** A lightweight object representing a location in the hierarchy (e.g. `net -> mlp -> dense1`). `graph.child('name')` appends to the path, ensuring every module has a unique address space.
* **The Params:** A flat, immutable dictionary holding all state, keyed by tuple paths (e.g. `('net', 'mlp', 'dense1', 'kernel')`). It supports simple partitioning for gradients or custom metadata.
* **The Rng:** `Params` maintains an Rng module such that when a module requests randomness, `Params` generates a unique, deterministic key via the Rng module and an *updated* `Params` structure with the Rng counter incremented. Modules can use a custom Rng module, to decouple their randomness from the `Params` randomness. See `Dropout` as an example.

## ⚠️ Gotchas

Because **blox** is functional, methods on `Params` return *new* instances rather than mutating in place. You must always reassign:

```python
# ✗ Wrong - result is discarded.
params.finalized()

# ✓ Correct - reassign to capture the new instance.
params = params.finalized()
```

The same applies to `Rng` accessor methods like `seed()`:

```python
# ✗ Wrong - params is not updated.
rng.seed(params, counter=0)

# ✓ Correct - capture the returned params.
params = rng.seed(params, counter=0)
```

Random keys generated under `jax.vmap` or `jax.shard_map` will be identical for each batch element/device unless you use `auto_fold_in_axes=True` (the default) on your `Rng` module. `axis_name` must also be specified for every `vmap` that wraps code using `Rng` to generate new random keys. See the [Batching & Parallel RNG](#-batching--parallel-rng-vmap--shard_map) section for more information.

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

1.  **Total transparency:** You know exactly what data your function touches.
2.  **JIT safety:** No global state means no side-effect leaks or tracer errors.
3.  **Maximum performance:** Zero overhead abstractions.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
