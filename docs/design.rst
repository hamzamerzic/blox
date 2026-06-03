Design Principles
=================

**blox** unlocks the full potential of JAX by embracing its functional nature instead of fighting it.

The Mental Model
----------------

The entire mental model fits in one line:

.. code-block:: python

   outputs, params = model(params, inputs)

Parameters go in, outputs and updated parameters come out. This is the standard pattern for `stateful computations in JAX <https://docs.jax.dev/en/latest/stateful-computations.html>`_. Because state flows explicitly through your code, all JAX transformations work out of the box.

Functional Purity
-----------------

Most JAX neural network libraries try to force Object-Oriented paradigms to make JAX feel like PyTorch, usually by introducing implicit global state, hidden contexts, or clever magic.

**blox** takes the opposite approach. Instead of hiding JAX's functional approach, it leans into it.

*   **Models are stateless:** A model is just a transformation definition.
*   **State is explicit:** Parameters are passed explicitly as arguments.
*   **No side effects:** Every function returns ``(outputs, params)``.

Core Abstractions
-----------------

We separate "structure" from "state".

Graph
~~~~~

A ``Graph`` object represents the hierarchical structure of your model (e.g., ``net -> mlp -> dense1``). It provides unique namespaces (paths) for parameters.

Key points:

* Paths are stored as tuples (e.g., ``('net', 'mlp', 'kernel')``), allowing any characters in names
* Use ``graph.child('name')`` or ``graph / 'name'`` to create child nodes
* Modules bind to graph nodes and cannot bind to root nodes (must use ``graph.child()``)
* Use ``graph.walk()`` to iterate over all descendant modules

Params
~~~~~~

A ``Params`` object is a flat, immutable container holding all state (weights, RNG keys, batch stats). It is keyed by the paths defined in the Graph.

Key points:

* All state is stored as ``Param`` objects with a ``trainable`` flag
* Use ``params.locked()`` after initialization to prevent accidental param creation
* Use ``params.split()`` to separate trainable from non-trainable state

Rng
~~~

An ``Rng`` module handles randomness. It is passed to modules on construction and stores its state (seed, counter) in the ``Params`` container.

Key points:

* Must be seeded first via ``rng.seed(params, seed=42)``
* Uses counter-based key generation: ``new_key = jax.random.fold_in(seed, counter)``
* Modules that need randomness (e.g., ``Dropout``) accept an ``Rng`` on construction
* In ``vmap``/``shard_map``, users must manually fold in axis indices for unique randomness (see below)

RNG in Parallel Contexts
~~~~~~~~~~~~~~~~~~~~~~~~

In ``vmap`` or ``shard_map``, all lanes share the same seed and counter, so they get identical random keys by default. To get unique randomness per lane, fold in a unique index.

Simplest approach - pass the index explicitly:

.. code-block:: python

   def apply_with_explicit_index(params, x, batch_idx):
     original_seed = rng.get_seed(params)
     folded_seed = jax.random.fold_in(original_seed, batch_idx)
     params = rng.seed(params, seed=folded_seed)
     out, params = dropout(params, x, is_training=True)
     params = rng.seed(params, seed=original_seed)  # Restore for replicated params
     return out, params

   # Pass jnp.arange(batch_size) as the batch indices
   jax.vmap(apply_with_explicit_index, in_axes=(None, 0, 0))(params, x, jnp.arange(4))

Or use ``jax.lax.axis_index`` with ``axis_name``. **This is the recommended approach** as it's more idiomatic:

.. code-block:: python

   def apply_with_axis_index(params, x):
     original_seed = rng.get_seed(params)
     folded_seed = jax.random.fold_in(original_seed, jax.lax.axis_index('batch'))
     params = rng.seed(params, seed=folded_seed)
     out, params = dropout(params, x, is_training=True)
     params = rng.seed(params, seed=original_seed)
     return out, params

   jax.vmap(apply_with_axis_index, in_axes=(None, 0), axis_name='batch')(params, x)

**Why restore the seed?** When params is replicated (``out_axes=None``), JAX requires all lanes to return identical pytrees. Since we run the same function in each lane, the counter increments identically everywhere. The seed is the only thing that differs (due to folding), so restoring it ensures params match.

**Init vs Runtime:** During init (params unlocked), don't fold: you want identical params. During runtime (params locked), do fold for unique randomness. Use ``params.is_locked`` to detect which mode.

Initialization with Sharded Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When initializing models with sharded parameters, **use** ``jax.jit`` **with** ``out_shardings`` **rather than** ``shard_map``.

Why ``shard_map`` is tricky for initialization:

* Different parameters may need different axes folded in (e.g., model axis but not batch axis)
* Multiple model axes mean different params have different sharding requirements
* Managing which axes to fold for which params becomes complex

Why ``jax.jit`` is better:

* Just specify ``out_shardings`` and JIT handles partitioning
* Use replicated RNG params during init
* JIT is smart about parameter placement during initialization

.. code-block:: python

   # RECOMMENDED: Initialize via JIT with out_shardings
   mesh = jax.make_mesh((4,), ('model',))
   params_sharding = ...  # Build from param metadata

   @jax.jit(out_shardings=params_sharding)
   def init():
       params = rng.seed(bx.Params(), seed=42)
       _, params = model(params, dummy_input)
       return params.locked()

   params = init()  # JIT handles sharding automatically

**Note:** These RNG patterns are a sharp edge in JAX that blox inherits. We hope JAX will provide better primitives in the future.

The Params Container
--------------------

The ``Params`` container holds **all** model state in one place: weights, RNG state, batch norm statistics, moving averages: everything.

**Why put RNG in Params?** In pure functional programming, randomness is state. If your dropout layer consumes a random key, that's a state change. By threading RNG through ``Params``, the signature ``outputs, params = model(params, inputs)`` tells the whole truth.

Every parameter is either:

.. list-table::
   :header-rows: 1

   * - Type
     - Examples
     - Gradients?
     - Updated during forward?
   * - **Trainable**
     - weights, embeddings
     - Yes
     - No
   * - **Non-trainable**
     - RNG counters, batch stats, EMA
     - No
     - Yes

The ``params.split()`` method separates these two categories for training.

JAX Compatibility
-----------------

**blox** works with all JAX transformations out of the box:

*   ``jax.jit`` - Just wrap and call
*   ``jax.grad`` - Use ``params.split()`` to separate trainable params
*   ``jax.vmap`` - Params can be replicated (``in_axes=None``) or batched
*   ``jax.shard_map`` - Use parameter metadata for sharding specs
*   ``jax.checkpoint`` - Works with the functional state pattern

No special wrappers or decorators are required.

Training Pattern
----------------

The standard training pattern uses ``split()`` and ``merge()``:

.. code-block:: python

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

Lazy Initialization
-------------------

**blox** uses lazy initialization:

1. Define model structure abstractly (no memory allocation)
2. Run a forward pass to trigger parameter creation
3. Lock params to prevent accidental creation during training

.. code-block:: python

   # Define structure.
   graph = bx.Graph('net')
   rng = bx.Rng(graph.child('rng'))
   model = MLP(graph.child('mlp'), hidden_size=128, output_size=10, rng=rng)

   # Initialize.
   params = bx.Params()
   params = rng.seed(params, seed=42)
   _, params = model(params, dummy_input)
   params = params.locked()

You can also use ``jax.eval_shape()`` to get parameter structure without allocating memory, useful for setting up sharding.

Why the Verbosity?
------------------

**blox chooses clarity over brevity.**

Most frameworks rely on implicit global state or thread-local contexts to hide parameters. While this saves a few keystrokes, it creates a "black box."

.. list-table::
   :header-rows: 1

   * - OOP-style Wrappers
     - **blox**
   * - ``out = layer(x)``
     - ``outputs, params = layer(params, inputs)``
   * - Implicit global state
     - Explicit state passing
   * - Opaque variable scopes
     - Explicit ``bx.Graph`` paths
   * - Custom ``vmap`` / ``jit`` wrappers
     - Standard ``jax.vmap`` / ``jax.jit``

By accepting slightly more verbose function signatures, you gain:

1. **Total transparency:** You know exactly what data your function touches.
2. **Full control:** No global state means no unknown side-effects.
3. **Maximum performance:** Zero overhead.

Decoupled Params and Graph
--------------------------

A key design principle in blox is the **clean separation between parameters and the model graph**. This is different from frameworks like Flax or Equinox where parameters are tightly coupled to modules.

**Why this separation?**

1. **No single module owns params.** Parameters are passed into functions, not stored in modules. This enables multiple models to operate on the same params without ownership conflicts.

2. **Avoids pytree complexity.** When modules are pytrees containing both static configuration and JAX arrays, you get finnicky behavior requiring magic handling and special wrappers. blox keeps a clean split: the ``Graph`` is purely static (Python objects describing structure), while ``Params`` is purely dynamic (JAX arrays ready for transformations).

3. **Graph is static, params are dynamic.** The graph describes *what* operations to perform. The params provide *what values* to use. This separation is maintained throughout execution; graph structure doesn't change at runtime, only param values do.

**Multiple models can share the same params:**

.. code-block:: python

   # Create two LSTM variants with the same parameter structure
   def create_lstm(is_static: bool):
       graph = bx.Graph('model')
       rng = bx.Rng(graph.child('rng'))
       lstm = bx.LSTM(graph.child('lstm'), hidden_size=64, rng=rng, is_static=is_static)
       return lstm

   lstm_static = create_lstm(is_static=True)    # Python loop (debuggable)
   lstm_dynamic = create_lstm(is_static=False)  # lax.scan (fast)

   # Initialize params once
   params = ...

   # Both models work with the same params!
   out_static, _ = lstm_static.apply(params, inputs, prev_state=state)
   out_dynamic, _ = lstm_dynamic.apply(params, inputs, prev_state=state)

**Use cases enabled by this design:**

1. **Actor/Learner pattern**: In RL, the Actor (collects data) and Learner (updates weights) can be separate models sharing params.

2. **Train/Eval pattern**: Training model (with dropout) and evaluation model (without) share the same params.

**Design recommendations:**

1. **Create static models with deliberate purpose.** Use factory functions to create model variants at construction time. Each model's purpose should be clear from how it's built, not from runtime modifications.

2. **Avoid runtime graph mutation.** While users have freedom to manipulate the graph, a clean design creates different models for different purposes rather than mutating one model at runtime.
