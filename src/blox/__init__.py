"""blox: A functional and lightweight neural network library for JAX.

Version: Use `blox.__version__` to get the installed version.

**blox** unlocks the full potential of JAX by embracing its functional nature.
Instead of forcing Object-Oriented paradigms (like PyTorch's `nn.Module`) onto
JAX, **blox** provides a minimal abstraction layer that keeps state explicit
and data flow transparent.

Key Concepts:
- **Explicit State:** Models are stateless transformations. All parameters are
  passed explicitly in a `Params` container. Functions return `(outputs, params)`.
- **Graph vs. Params:** `Graph` defines the hierarchical structure (naming/paths),
  while `Params` holds the actual state (weights, RNG keys).
- **Lazy Initialization:** You define the model structure once. Parameters are
  materialized only when you run a forward pass with `params` containing an RNG.
- **JAX Compatibility:** Works seamlessly with `jax.jit`, `jax.vmap`, `jax.grad`,
  and `jax.shard_map`.
- **Structural RNG:** Randomness is handled via `Params`. `Rng` automatically folds
  in `vmap`/`shard_map` axes to split keys across devices/batches.

Gotchas:
- **Purity:** Modules must be pure. Do not store state in `self`. Use `get_param`.
- **Initialization:** You must call `finalized()` on params after the first pass
  to prevent accidental creation of new parameters during training/inference.
- **Sequence Models:** RNNs (`LSTM`, `GRU`) process entire sequences by default
  using `jax.lax.scan`. Use `__call__` for single-step processing.
"""

from importlib.metadata import version as _version

from .blocks import (
    GRU,
    LSTM,
    BatchNorm,
    Conv,
    ConvTranspose,
    Dropout,
    Embed,
    GRUState,
    LayerNorm,
    Linear,
    LSTMState,
    RMSNorm,
    Sequential,
    avg_pool,
    max_pool,
    min_pool,
)
from .interfaces import (
    Graph,
    Module,
    Param,
    Params,
    RecurrenceBase,
    Rng,
    SequenceBase,
    dynamic_scan,
    static_scan,
)
from .visualize import display

__version__ = _version('jax-blox')

__all__ = [
    # Core.
    'Graph',
    'Module',
    'Param',
    'Params',
    'Rng',
    'display',
    # Layers.
    'Embed',
    'Linear',
    'Sequential',
    'Conv',
    'ConvTranspose',
    'Dropout',
    'LayerNorm',
    'RMSNorm',
    'BatchNorm',
    # Pooling.
    'max_pool',
    'min_pool',
    'avg_pool',
    # Sequence processing.
    'SequenceBase',
    'RecurrenceBase',
    'LSTM',
    'LSTMState',
    'GRU',
    'GRUState',
    'static_scan',
    'dynamic_scan',
]
