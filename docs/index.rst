:hide-toc:

.. rst-class:: landing-title

blox
====

.. rst-class:: landing-tagline

A functional and lightweight neural network library for JAX.

.. raw:: html

   <p class="landing-badges">
     <a href="https://github.com/hamzamerzic/blox/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
     <a href="https://pypi.org/project/jax-blox/"><img src="https://img.shields.io/pypi/v/jax-blox.svg" alt="PyPI"></a>
     <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python 3.11+">
     <img src="https://img.shields.io/badge/jax-0.10+-green" alt="JAX 0.10+">
   </p>

----

**blox** unlocks the full potential of JAX by embracing its functional
nature instead of fighting it.

JAX gives you composable transformations that let you write math and have
it run blazingly fast on any hardware. **blox** is a thin layer on top
that keeps all of that power accessible while giving you just enough
structure to organize your neural networks.

The entire mental model fits in one line:

.. code-block:: python

   outputs, params = model(params, inputs)

Parameters go in, outputs and updated parameters come out. Because state
flows explicitly through your code, all JAX transformations —
``jax.jit``, ``jax.grad``, ``jax.vmap``, ``jax.checkpoint`` — work out
of the box. No wrappers, no decorators, no surprises.

Who is blox for?
----------------

* **Learners.** There is no "framework magic" to learn. What you see is
  what you get — the best way to understand how neural networks actually
  work at the JAX level.
* **Practitioners.** If you're tired of fighting frameworks that hide
  important details, blox gives you complete transparency. Whether you're
  building custom training loops, implementing novel architectures, or
  scaling up, you have direct access to the full execution stack.

Installation
------------

Since blox uses JAX, check the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
for your specific hardware (CPU / GPU / TPU). You will need Python 3.11 or
later.

Install the latest release from PyPI:

.. code-block:: bash

   pip install jax-blox

Or install the development version from source:

.. code-block:: bash

   pip install git+https://github.com/hamzamerzic/blox.git

Quickstart
----------

Define a layer:

.. code-block:: python

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

Wire it up and run a forward pass:

.. code-block:: python

   graph = bx.Graph('net')
   rng = bx.Rng(graph.child('rng'))
   model = Linear(graph.child('linear'), output_size=10, rng=rng)

   params = bx.Params()
   params = rng.seed(params, seed=42)
   outputs, params = model(params, jnp.ones((4, 784)))
   params = params.locked()  # Lock to prevent accidental param creation.

JIT, grad, vmap — they all just work, because the function signature
already tells the whole truth.

.. code-block:: python

   outputs, params = jax.jit(model)(params, inputs)

Continue with the :doc:`MNIST tutorial <mnist_tutorial>` for an
end-to-end example, or read the :doc:`design notes <design>` to see how
the pieces fit together.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   mnist_tutorial
   rnn_tutorial

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Key Concepts

   design

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Advanced Topics

   checkpointable_training
   lora_example
   sharp_bits

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   api
