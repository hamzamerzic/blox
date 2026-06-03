.. meta::
   :description: How blox compares to Equinox and Flax NNX: where each library departs from JAX's explicit, no-magic core, and how blox keeps the model graph and parameter tree as separate objects so every JAX transform applies directly.
   :keywords: JAX, neural network library, Equinox alternative, Flax NNX alternative, pytree, filtered transformations, reference semantics

Comparison: Equinox & Flax NNX
==============================

JAX already ships a strong abstraction: composable transformations over pure
functions, with `state threaded explicitly through function signatures
<https://docs.jax.dev/en/latest/stateful-computations.html>`_. The question a
neural-network library answers is *how much of that stays visible*. Equinox and
Flax NNX are both mature, well-built libraries that take different routes, and
both make simple things easy while adding cognitive overhead as models get
complicated. This page documents where each one departs from JAX's "explicit,
no hidden state, no magic" core, and how blox avoids those departures by
construction.

A note on fairness up front. Neither library is bad. Equinox stays close to
JAX and is explicitly designed to "feel like (and be fully compatible with) the
main JAX library itself." NNX's reference semantics make model surgery and
inspection genuinely easy, and it has a large community and a strong
development team behind it. The argument below is about *where the abstraction
leaks*, not about whether these libraries are good.

Equinox: modules as pytrees, plus a filtering layer
---------------------------------------------------

Equinox keeps the model *as* a PyTree, which is close to JAX in spirit. But
because the module pytree mixes array leaves with arbitrary Python (non-array)
leaves, plain ``jax.jit`` / ``jax.grad`` don't apply directly to a realistic
model. From the `Equinox FAQ <https://docs.kidger.site/equinox/faq/>`_:

   "This error happens because a model, when treated as a PyTree, may have
   leaves that are not JAX types (such as functions). It only makes sense to
   trace arrays. … Instead of ``jax.jit``, use ``equinox.filter_jit``. Likewise
   for other transformations."

So in practice you adopt a parallel family of *filtered* transformations
(``filter_jit`` / ``filter_grad`` / ``filter_vmap`` / ``filter_pmap``), and the
underlying ceremony is ``partition`` → transform → ``combine``. You also learn
which default filter applies: ``is_array`` (all arrays) vs ``is_inexact_array``
(only float arrays get gradients), since ``filter_grad`` silently returns
``None`` gradients for non-float leaves.

Coupling state to the module pytree has real foot-guns. With ``BatchNorm``, the
module's flattened pytree form changes across iterations (`equinox#238
<https://github.com/patrick-kidger/equinox/issues/238>`_), and tied/shared
layers diverge under value semantics; the FAQ notes that after some gradient
updates "you'll find that ``self.linear1`` and ``self.linear2`` are now
different." Stateful layers must be threaded by hand: ``x, state =
self.norm(x, state)``.

Flax NNX: a separate reimplementation of JAX's transforms
---------------------------------------------------------

Flax NNX goes the other way, with mutable, PyTorch-style module objects and
reference semantics. Because JAX transforms can't operate on mutable
reference-semantic objects, NNX ships its own version of essentially the entire
JAX transform surface. From the `NNX Transformations guide
<https://flax.readthedocs.io/en/latest/guides/transforms.html>`_:

   "JAX transformations operate on pytrees of ``jax.Array``\\ s and abide by
   value semantics. This presents a challenge for Flax NNX, which represents
   ``nnx.Module``\\ s as regular Python objects that follow reference
   semantics."

The result is a full parallel transform suite (``nnx.jit``, ``nnx.grad``,
``nnx.value_and_grad``, ``nnx.vmap``, ``nnx.pmap``, ``nnx.scan``, ``nnx.remat``,
``nnx.cond``, ``nnx.while_loop``, and more), described in the
`NNX basics <https://flax.readthedocs.io/en/latest/nnx_basics.html>`_ as
"supersets of their equivalent JAX counterparts." It is a divergent
reimplementation, not a thin wrapper. ``nnx.scan`` "(consciously) deviates from
``jax.lax.scan``." Underneath sits a ``Module`` / ``State`` / ``GraphDef``
system (NNX is a graph, not a pytree, by design), and crossing into a JAX
transform requires a ``split`` / ``merge`` ceremony.

This is a real maintenance surface and a known leaky abstraction. Mutation
"must be used with care because it can clash with JAX's underlying assumptions"
(changing graph structure inside ``nnx.jit`` "causes continuous
recompilations"), and the maintainers' own long-term plan is to make NNX
`implement the pytree protocol
<https://github.com/google/flax/discussions/4736>`_ specifically so it can be
"used with raw JAX transformations and other libraries", i.e. to stop
diverging from raw JAX.

The pattern: simple is easy, complex gets steep
-----------------------------------------------

Both libraries make the toy case easy and add overhead once you leave it.
Equinox's surface grows with custom initialisation, parameter surgery, and
scan-over-layers ("The above code probably seems a bit complicated!", the
Equinox `tricks <https://docs.kidger.site/equinox/tricks/>`_ page). NNX's
mutation (the thing that buys the PyTorch-like ergonomics) is also the thing
that introduces a new error class (``Inconsistent aliasing detected``) and the
silent-recompile trap. In both cases the cost lands at the edges, where you
cross a transform boundary or step outside the happy path.

How blox avoids both boundaries
-------------------------------

blox keeps ``Graph`` (static Python describing structure) and ``Params``
(dynamic JAX arrays) as *two separate objects*. There is no mixed-leaf pytree
to filter (as in Equinox) and no mutable graph to split and merge (as in NNX).
``jax.jit``, ``jax.grad``, ``jax.vmap``, and ``jax.checkpoint`` apply to a plain
function with no ``filter_*`` and no ``nnx.*`` wrapper in the way. See
:doc:`design` for the full rationale behind the params/graph split.

Randomness is the one sharp edge every JAX library inherits, and the three
handle it differently. Equinox threads ``jax.random`` keys through your
functions by hand. NNX hides them inside a stateful ``nnx.Rngs`` object whose
keys live in the graph and need "extra tricks with ``nnx.vmap``" (per the Flax
`randomness guide <https://flax.readthedocs.io/en/latest/guides/randomness.html>`_)
to behave correctly under transforms, namely ``nnx.split_rngs`` and
``nnx.StateAxes`` when you ``vmap`` or ``scan``. blox keeps JAX's own counter-based ``fold_in``
pattern and surfaces it explicitly rather than wrapping it (see
:doc:`sharp_bits` / the RNG notes). The sharp edge is JAX's, and so is
everything you learn working around it.

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 25

   * -
     - Equinox
     - Flax NNX
     - **blox**
   * - Calls JAX transforms directly
     - Mostly: non-array leaves force ``filter_*``
     - No: ``nnx.jit`` / ``nnx.scan`` / … reimplementations
     - **Yes: unwrapped ``jax.jit`` / ``grad`` / ``vmap`` / ``checkpoint``**
   * - Boundary ceremony
     - ``partition`` / ``combine``, filter specs
     - ``nnx.split`` / ``nnx.merge`` (State / GraphDef)
     - **None: ``params`` is already a clean array pytree**
   * - Where state lives
     - In the module pytree
     - In mutable ``Module`` instances (a graph)
     - **In a separate ``Params`` container**
   * - Library-specific transforms to maintain
     - filtered transforms
     - a full parallel transform suite
     - **Zero**
   * - Randomness
     - manual ``jax.random`` key threading
     - stateful ``nnx.Rngs`` in the graph (``split_rngs`` / ``StateAxes`` for ``vmap`` / ``scan``)
     - **JAX's own ``fold_in`` pattern, surfaced**
   * - Main foot-gun
     - value-semantics surprises (shared layers, BatchNorm tree drift)
     - mutation vs ``jit`` (silent recompiles, aliasing rules)
     - JAX PRNG folding: surfaced, not hidden

Both Equinox and NNX are mature and a great fit for many projects: Equinox if
you like "the model is a pytree," NNX if mutable PyTorch-style objects feel
natural. **blox makes a different bet: rather than building a framework on top
of JAX, it grows directly out of JAX's own philosophy: explicit state, pure
functions, no hidden magic.** The graph and the parameters stay separate, every
transformation is the real ``jax.*`` one, and the randomness is JAX's own. What
you learn using blox is JAX itself, so your understanding and your code keep
paying off as the ecosystem moves, with nothing library-specific standing in
the way.

Sources
-------

- Equinox: `FAQ <https://docs.kidger.site/equinox/faq/>`_ ·
  `Transformations API <https://docs.kidger.site/equinox/api/transformations/>`_ ·
  `Tricks / surgery <https://docs.kidger.site/equinox/tricks/>`_ ·
  `#238 BatchNorm flattened-form drift <https://github.com/patrick-kidger/equinox/issues/238>`_
- Flax NNX: `NNX Basics <https://flax.readthedocs.io/en/latest/nnx_basics.html>`_ ·
  `Transformations guide <https://flax.readthedocs.io/en/latest/guides/transforms.html>`_ ·
  `transforms API <https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html>`_ ·
  `Discussion #4736 (pytree-protocol plan) <https://github.com/google/flax/discussions/4736>`_ ·
  `FLIP #4107 <https://github.com/google/flax/issues/4107>`_
