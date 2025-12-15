Design Principles
=================

**blox** unlocks the full potential of JAX by embracing its functional nature instead of fighting it.

Functional Purity
-----------------

Most JAX neural network libraries try to force Object-Oriented paradigms to make JAX feel like PyTorch, usually by introducing implicit global state, hidden contexts, or clever magic.

**blox** takes the opposite approach. Instead of hiding JAX's functional approach, it leans into it.

*   **Models are stateless:** A model is just a transformation definition.
*   **State is explicit:** Parameters are passed explicitly as arguments.
*   **No side effects:** Every function returns `(outputs, params)`.

The Graph & The Params
----------------------

We separate "structure" from "state".

The Graph
~~~~~~~~~

A ``Graph`` object represents the hierarchical structure of your model (e.g., `net -> mlp -> dense1`). It provides unique namespaces (paths) for parameters.

The Params
~~~~~~~~~~

A ``Params`` object is a flat, immutable container holding all state (weights, RNG keys, batch stats). It is keyed by the paths defined in the Graph.

Key Features
------------

*   **Native JAX Compatibility:** Works with ``jax.jit``, ``jax.grad``, ``jax.vmap``, ``jax.shard_map`` out of the box.
*   **Lazy Initialization:** Define structure abstractly, run a forward pass to materialize parameters.
*   **Structural RNG:** Randomness is handled as part of the ``Params`` structure. Getting a new key returns an updated ``Params`` object.
