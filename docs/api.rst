API Reference
=============

.. module:: blox

Core Interfaces
---------------

.. autoclass:: Graph
   :members:

.. autoclass:: Module
   :members:

.. autoclass:: Params
   :members:

.. autoclass:: Param
   :members:

.. autoclass:: Rng
   :members:

Layers
------

.. module:: blox.blocks

.. autoclass:: blox.Embed
   :members:

.. autoclass:: blox.Linear
   :members:



.. autoclass:: blox.Sequential
   :members:

.. autoclass:: blox.Conv
   :members:

.. autoclass:: blox.ConvTranspose
   :members:

.. autoclass:: blox.Dropout
   :members:

.. autoclass:: blox.LayerNorm
   :members:

.. autoclass:: blox.RMSNorm
   :members:

.. autoclass:: blox.BatchNorm
   :members:

Pooling
-------

.. autofunction:: blox.max_pool
.. autofunction:: blox.min_pool
.. autofunction:: blox.avg_pool

Sequence Processing
-------------------

.. autoclass:: blox.SequenceBase
   :members:

.. autoclass:: blox.RecurrenceBase
   :members:

.. autoclass:: blox.LSTM
   :members:

.. autoclass:: blox.LSTMState
   :members:

.. autoclass:: blox.GRU
   :members:

.. autoclass:: blox.GRUState
   :members:

.. autofunction:: blox.static_scan
.. autofunction:: blox.dynamic_scan

Visualization
-------------

.. module:: blox.visualize

.. autofunction:: blox.display
