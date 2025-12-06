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

**blox** embraces JAX's functional paradigm without compromise.

It provides a minimal, object-oriented layer solely for code organization, while enforcing strictly functional state management and explicit data flow. By removing the "magic" found in other frameworks—such as context managers, thread-local storage, and global state—Blox ensures that your code remains side-effect free, transparent, and trivially compatible with JAX's powerful transformations (`jit`, `grad`, `vmap`).

It is designed for users who want the structural benefits of a library like Flax, but with the raw transparency and zero-overhead performance of pure JAX.

## ⚡ Core Principles

* **Functional purity:** Models are stateless transformations. Parameters and RNG state are passed explicitly as arguments (`params`), never stored in `self`.
* **Explicit data flow:** There is no hidden global context. You can trace the path of every tensor by reading the function signature alone.
* **Structural RNG:** Random keys are derived deterministically from the graph structure, eliminating the need to manually pass keys through every layer ("refactoring hell") while maintaining strict functional purity.
* **Visualizable:** Includes out-of-the-box integration with **Treescope** for rich, interactive visualization of model architectures and parameter hierarchies.

## 📦 Installation

```bash
git clone [https://github.com/hamzamerzic/blox.git](https://github.com/hamzamerzic/blox.git)
cd blox
pip install -e .
```

## 🚀 Quick Start

In blox, a module defines a structural hierarchy (`__init__`) and a pure mathematical function (`__call__`).

### Define your layers

Notice the signature: `params` carries the state (weights + RNG), while `x` is the data.

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
      x: jax.Array,
  ) -> tuple[jax.Array, bx.Params]:
    # Request parameters explicitly from the container.
    # The RNG key is automatically derived from the graph path.
    w_shape = (x.shape[-1], self.output_size)
    w, params = self.get_param(
        params, 'w', w_shape, jax.nn.initializers.glorot_uniform()
    )
    b_shape = (self.output_size,)
    b, params = self.get_param(
        params, 'b', b_shape, jax.nn.initializers.zeros
    )
    return x @ w + b, params
```

### Composition

We use the `Graph` object to define the immutable hierarchy of the model.

```python
class CustomMLP(bx.Module):

  def __init__(
      self,
      graph: bx.Graph,
      hidden_size: int,
      output_size: int,
  ) -> None:
    super().__init__(graph)
    # Define the graph structure.
    self.l1 = CustomLinear(graph.child('dense1'), hidden_size)
    self.l2 = CustomLinear(graph.child('dense2'), output_size)

  def __call__(
      self,
      params: bx.Params,
      x: jax.Array,
  ) -> tuple[jax.Array, bx.Params]:
    # Chain the functional transformations.
    x, params = self.l1(params, x)
    x = jax.nn.relu(x)
    x, params = self.l2(params, x)
    return x, params
```

### Initialization & Visualization

blox cleanly separates the "Initialization phase" (where the graph is traversed to create parameters) from the "Runtime phase" (where parameters are trained).

```python
# Structure.
graph = bx.Graph('net')
model = CustomMLP(graph.child('mlp'), hidden_size=32, output_size=1)

# Data and state.
x = jnp.ones((1, 10))
# Seed the Params container with a master RNG key (or integer seed).
params = bx.Params(42)

# Initialization pass.
y, params = model(params, x)

# Finalize initialization (locks container against accidental creation).
params = params.finalize()

# Visualize interactive structure in notebooks.
bx.display(graph, params)
```

**Output:**
```text
net: Graph # Param: 385 (1.5 KB)(
  rng=Param[N](
    shape=(2,),
    dtype=object,
    metadata={'tag': 'rng'},
    value=(<jax.Array...>, <jax.Array...>)
  ),
  mlp=CustomMLP # Param: 385 (1.5 KB)(
    hidden_size=32,
    output_size=1,
    dense1=CustomLinear # Param: 352 (1.4 KB)(
      output_size=32,
      w=Param[T](value=<jax.Array...>),
      b=Param[T](value=<jax.Array...>)
    ),
    dense2=CustomLinear # Param: 33 (132.0 B)(
      output_size=1,
      w=Param[T](value=<jax.Array...>),
      b=Param[T](value=<jax.Array...>)
    )
  )
)
```

## ⚡ Training (JIT & Gradients)

Since `Params` contains non-differentiable state (like the RNG counter), we must **partition** the parameters before taking gradients.

```python
@jax.jit
def train_step(params, x, y):
  # Partition split differentiable weights from RNG/Frozen state.
  trainable, non_trainable = params.partition(lambda p, v: v.trainable)

  def loss_fn(t):
    # Merge parts to run the model.
    full_params = t.merge(non_trainable)
    pred, _ = model(full_params, x)
    return jnp.mean((pred - y) ** 2)

  # Differentiate only w.r.t. trainable weights.
  grads = jax.grad(loss_fn)(trainable)

  # Apply SGD to trainable weights.
  new_trainable = jax.tree.map(lambda w, g: w - 0.01 * g, trainable, grads)
  
  # Return merged state.
  return new_trainable.merge(non_trainable)
```

## 🧠 Under the Hood (No Magic)

blox is designed to be fully transparent. The only "abstraction" is the automated handling of paths and keys to keep your code clean. Here is exactly what happens inside.

**The Graph**

This is simply a string builder. When you call `graph.child("dense1")`, it appends strings to create a unique path ID like `"net/mlp/dense1"`. This ensures every parameter has a unique, readable name.

**The Params**

This is a dictionary wrapper mapping path strings (e.g., `"net/mlp/dense1/w"`) to `Variable` objects. A `Variable` holds the JAX array, a `trainable` boolean flag, and a metadata dictionary. It is a standard JAX Pytree, so `jax.jit` treats it like a normal structure.

**The RNG**

Instead of passing `key` arguments manually through every function, `Params` holds a master seed and a counter. When a layer needs randomness, blox uses `jax.random.fold_in(master_key, counter)` to generate a deterministic, unique key for that specific call, and increments the counter. This is mathematically safe, parallel-friendly, and maintains functional purity.

## ⚖️ Why blox?

**blox chooses clarity over brevity.**

| Standard Frameworks | blox |
| :--- | :--- |
| `out = layer(x)` | `out, params = layer(params, x)` |
| Implicit context managers | Explicit state passing |
| Hidden variable scopes | Explicit `bx.Graph` definitions |

By accepting slightly more verbose function signatures, you gain:
1.  **Total transparency:** You know exactly what data your function touches.
2.  **JIT safety:** It is impossible to leak tracers or capture side-effects, as there is no global state.
3.  **Performance:** The library compiles down to the exact same XLA kernels as raw JAX code.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.