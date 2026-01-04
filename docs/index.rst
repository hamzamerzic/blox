Welcome to blox's documentation!
================================

**blox** is a functional and lightweight neural network library for JAX.

The entire mental model fits in one line:

.. code-block:: python

   outputs, params = model(params, inputs)

Parameters go in, outputs and updated parameters come out. Because state flows explicitly through your code, all JAX transformations work out of the box. No wrappers, no decorators, no surprises.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   mnist_tutorial
   rnn_tutorial

.. toctree::
   :maxdepth: 2
   :caption: Key Concepts

   design

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   checkpointable_training
   lora_example
   sharp_bits

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
