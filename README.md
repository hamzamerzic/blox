<div align="center">
  <img src="images/logo.png" width="400" alt="blox logo">
  
  <h1>blox</h1>
  
  <p>
    <strong>A lightweight, strictly functional neural network library for JAX.</strong>
  </p>

  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="blox is released under the MIT license">
  </a>
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/jax-0.4+-green" alt="JAX 0.4+">

</div>

---

**blox** embraces JAX's functional paradigm without the headache.

It provides a minimal, object-oriented layer solely for organizing your code, while strictly enforcing functional state management and explicit data flow. By stripping away the "magic" found in other frameworks—like implicit context managers, thread-local storage, and global state—**blox** ensures your code remains side-effect free, transparent, and trivially compatible with JAX's powerful transformations.

**No wrappers needed.** Because there is no hidden state, `jax.jit`, `jax.grad`, and `jax.vmap` work right out of the box.

## ⚡ Core Principles

* **Functional purity:** Models are just stateless transformations. Parameters and RNG state are passed explicitly as arguments (`params`), never stored in `self`.
* **Explicit data flow:** No hidden global context. You can trace the path of every single tensor just by reading the function signature.
* **Structural RNG:** Random keys are derived deterministically from the graph structure. Say goodbye to manually threading keys through every layer ("refactoring hell"); blox handles the math while keeping your functions pure.
* **Visualizable:** Comes with out-of-the-box **Treescope** integration for beautiful, interactive visualization of your model's architecture and parameters.

## 📦 Installation

```bash
git clone https://github.com/hamzamerzic/blox.git
cd blox
pip install -e .
```

## 🚀 Quick Start

In blox, a module is just a structural container (`__init__`) and a pure mathematical function (`__call__`).

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
    # Request parameters explicitly from the container.
    # The RNG key is automatically derived from the graph path.
    w_shape = (inputs.shape[-1], self.output_size)
    w, params = self.get_param(
        params, 'w', w_shape, jax.nn.initializers.glorot_uniform()
    )
    b_shape = (self.output_size,)
    b, params = self.get_param(
        params, 'b', b_shape, jax.nn.initializers.zeros
    )
    return inputs @ w + b, params
```

### Composition & Dependency Injection

Because **blox** modules are just standard Python objects, dependency injection is a breeze.

Instead of hardcoding layers, you can create modules outside and pass them in. This changes the graph hierarchy: the injected module keeps its original path (it's a "sibling"), while internal layers become children.

```python
class CustomMLP(bx.Module):

  def __init__(
      self,
      graph: bx.Graph,
      hidden_size: int,
      # Inject a pre-built module instance.
      output_projection: bx.Module,
  ) -> None:
    super().__init__(graph)
    # Internal layer: We create it here, so it lives in our scope.
    self.hidden_proj = CustomLinear(graph.child('hidden'), hidden_size)
    
    # Injected layer: It was created outside, so we just store the reference.
    self.output_projection = output_projection

  def __call__(
      self,
      params: bx.Params,
      inputs: jax.Array,
  ) -> tuple[jax.Array, bx.Params]:
    # Chain the functional transformations.
    x, params = self.hidden_proj(params, inputs)
    x = jax.nn.relu(x)
    
    # The output projection knows where to find its own params in the container.
    outputs, params = self.output_projection(params, x)
    return outputs, params
```

### Initialization & Visualization

We cleanly separate the "Initialization phase" (traversing the graph to create parameters) from the "Runtime phase" (training the parameters).

```python
# Define the structure (Wiring).
graph = bx.Graph('net')

# Create the output layer explicitly at the root level ('net/readout').
readout = CustomLinear(graph.child('readout'), output_size=1)

# Pass it into the MLP. 
# The MLP lives at 'net/mlp', but it uses 'readout' which lives at 'net/readout'.
model = CustomMLP(graph.child('mlp'), hidden_size=32, output_projection=readout)

# Create Data and Seed.
inputs = jnp.ones((1, 10))
params = bx.Params(seed=42)

# Initialization Pass.
# We run the model once to populate the params container.
outputs, params = model(params, inputs)

# Finalize initialization.
# This prevents further changes to the parameter structure (like accidentally 
# adding new parameters after initialization).
params = params.finalize()

# Visualize.
bx.display(graph, params)
```

**Output:**
Notice how `readout` and `mlp` are siblings in the graph, while `hidden` is nested inside `mlp`.

```text
net: Graph # Param: 385 (1.5 KB)(
  rng=Param[N](
    shape=(2,),
    dtype=object,
    metadata={'tag': 'rng'},
    value=(<jax.Array...>, <jax.Array...>)
  ),
  readout=CustomLinear # Param: 33 (132.0 B)(
    output_size=1,
    w=Param[T](value=<jax.Array...>),
    b=Param[T](value=<jax.Array...>)
  ),
  mlp=CustomMLP # Param: 352 (1.4 KB)(
    hidden=CustomLinear # Param: 352 (1.4 KB)(
      output_size=32,
      w=Param[T](value=<jax.Array...>),
      b=Param[T](value=<jax.Array...>)
    )
  )
)
```

## ⚡ Training (JIT & Gradients)

The `Params` container holds *everything*: weights, biases, RNG state, batch norm statistics, and EMA moving averages.

When training, we usually want to differentiate with respect to weights, but we still need to update the non-trainable state (like the RNG counter or batch statistics). **blox** makes this partitioning explicit.

```python
@jax.jit
def train_step(params, inputs, targets):
  # Split params into two sets.
  # Trainable: weights, biases (we want gradients for these).
  # Non-trainable: RNG, batch stats, EMA (we just want the updated values).
  trainable, non_trainable = params.split_trainable()

  def loss_fn(t, nt):
    # Merge parameters to run the forward pass.
    predictions, new_params = model(t.merge(nt), inputs)

    # Calculate the loss.
    loss = jnp.mean((predictions - targets) ** 2)

    # Extract the updated non-trainable state to pass it out.
    _, new_non_trainable = new_params.split_trainable()
    return loss, new_non_trainable

  # Calculate gradients and capture the auxiliary state.
  grads, new_non_trainable = jax.grad(loss_fn, has_aux=True)(
      trainable, non_trainable
  )

  # Update the trainable weights using SGD.
  new_trainable = jax.tree.map(lambda w, g: w - 0.01 * g, trainable, grads)

  # Merge the updated weights with the updated non-trainable state.
  return new_trainable.merge(new_non_trainable)
```

## 🧠 Under the Hood (No Magic)

blox is designed to be fully transparent. The "abstraction" is really just automated path handling to keep your code clean and your state pure.

**The Graph**
This acts as a **Path Builder**. It is a lightweight object that represents a location in the model hierarchy (e.g., `net/mlp/dense1`). When you call `graph.child('name')`, it appends to the path. This ensures that every module has a unique address space for its variables.

**The Params**
This is a **Secure Vault**. It holds all weights, biases, and RNG states in a single, flat, immutable dictionary keyed by the paths generated by the Graph (e.g., `"net/mlp/dense1/w"`). It provides methods to partition state (for gradients) and merge it back (for updates).

**The RNG**
Handling randomness in pure functional programming can be painful. Instead of manually threading `key` arguments through every single layer, `Params` maintains a master key and a counter.
* When a module needs randomness (e.g., initialization or dropout), it asks `Params` for a key.
* `Params` uses `jax.random.fold_in(master_key, counter)` to generate a deterministic, unique key for that specific call.
* It increments the counter and returns a *new* `Params` object.
* This guarantees that your model is mathematically reproducible and parallel-safe without the boilerplate.

## ⚖️ Why blox?

**blox chooses clarity over brevity.**

Most frameworks rely on implicit global state or thread-local contexts to save you from passing arguments. This works great until you need to debug a side-effect, use a transformation the framework wasn't designed for, or inspect the state mid-execution.

| Standard Frameworks | blox |
| :--- | :--- |
| `out = layer(x)` | `out, params = layer(params, inputs)` |
| Implicit global context | Explicit state passing |
| Hidden variable scopes | Explicit `bx.Graph` paths |
| Custom `jit` / `vmap` wrappers | Standard `jax.jit` / `jax.vmap` |

By accepting slightly more verbose function signatures, you gain:
1.  **Total transparency:** You know exactly what data your function touches.
2.  **JIT safety:** It is impossible to leak tracers or capture side-effects, as there is no global state.
3.  **Performance:** The library compiles down to the exact same XLA kernels as raw JAX code.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.