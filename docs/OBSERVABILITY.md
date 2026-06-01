# The observability problem (and what this project actually changes)

This is the "why" behind the architecture. It answers one objection in particular:
*every language model maps tokens → a next-token distribution, and you can read
that — so what is the interpretability problem?*

## The boundary is not the problem

Every LM — a bigram table, GPT-2, GPT-4 — maps input tokens to a distribution
over next tokens, and you can always read the input tokens and the output
distribution. The endpoints are observable by definition. "`a` → {`b`, `c`, …}
is readable" is true universally and was never the hard part.

## The problem is the *mechanism*, not the I/O

Between the input token and the output distribution sits the actual computation:
in a real LM, dozens of layers and billions of parameters of vector
transformations. The questions that matter are about *that*:

- What intermediate concepts/words were activated and combined along the way?
- *Why* this prediction and not another — what is the chain of steps?
- What would you have to change *inside* to change the answer? (causality)

For a normal transformer you can read `cat → sat (p=0.3)` at the boundary, but
you **cannot** read why: the intermediate states are opaque `d`-dim vectors in
per-layer rotated bases, and the weight matrices map opaque-space → opaque-space
(no single weight means anything). Recovering meaning requires *post-hoc,
approximate* machinery layered on top — the logit lens, linear probes, sparse
autoencoders, activation/path patching. That opacity in the middle, at scale, is
the interpretability problem.

## What vocabulary-space changes

Because `hidden_dim == vocab_size` and there is no embedding or output
projection, the *middle* becomes readable in the same units as the boundary:

1. **The read is exact, not approximate.** The hidden state *is* a distribution
   over words; reading it is the identity map, not a learned projection (the
   logit lens, but lossless and free).
2. **The weights are interpretable, not just the activations.** v8's interaction
   matrix `W = U@Vᵀ + diag(d)` is literally a "word *i* → word *j*" table you read
   straight off the parameters — the learned *rule*, not just the runtime state.
3. **One shared, fixed basis everywhere → linking is reading, not patching.**
   Every state and every weight live in the same vocab basis, so you compose them
   directly ("these words active at step 3 → through `W` → these words at step 4")
   instead of reconstructing cross-layer links with activation patching.

Concretely, `observe trace` on a trained v8 shows the prediction *assembling*:
bytes → morphemes (`ing`, `ed`, `ly`) → words → `the / a / to`. That is reading
the mechanism, step by step — not just the endpoints.

So existing interpretability is **post-hoc reverse-engineering of an opaque
model**; this is **interpretability by construction** — meaning is baked into the
coordinate system, so understanding is *reading* instead of *reconstructing*.

## The honest catch

Two limits keep this from being a free lunch:

- **Readable ≠ faithful.** A state being legible does not mean it *causes* the
  output. Measured on real checkpoints (`observe causality`): v8's active word is
  **load-bearing** (boosting it flips the prediction), but v12's probed mid-state
  is **not** (perturbing it changes nothing downstream — readable but
  epiphenomenal at that site). Readability is necessary, not sufficient; causal
  verification is the lynchpin, not an afterthought.
- **Readable-middle is only interesting if the model is hard.** Simple models are
  already interpretable — a bigram table *is* a readable "word → word" map. In the
  controlled demo, v8 recovered a planted bigram at 100%, which proves the
  mechanism works **but is near-trivial** on its own. The contribution only counts
  if the model does something a bigram cannot — multi-hop, contextual,
  compositional prediction — **and stays readable through it.**

## Why the goal is "observability at *no cost*"

The two limits above force the real objective: a model that is **both** as good as
an opaque baseline **and** observable. Observability bought by crippling the model
proves nothing (you've just built a readable bad model); the interesting result is
readability that survives competitive performance. That is why this project
measures performance and observability *jointly*, and treats the ~0.82 bpb gap to
the opaque `v13` baseline as a gap to *eliminate*, not a price to pay.

One more limit worth stating plainly: this is a way to *build* observable models,
not to crack open an existing opaque one. It cannot be applied post-hoc to GPT-4 —
the architecture has to be designed this way. The bet is that capable models can
be observable *from the start*.

## Early evidence that it can be free (2026-06)

Measuring both axes (bits-per-byte for performance, `observe coverage` for
observability — the fraction of readable active-word sites that are causally
load-bearing), scaling v8's interaction **width** improves *both at once*, and the
trend holds across the whole sweep (600s each on 1× H100; train ≈ val throughout,
i.e. **no memorization at any rank**):

| v8 width | params | bpb ↓ | cov @τ.02 | @τ.05 | @τ.10 | median Δlogits |
|---|---|---|---|---|---|---|
| rank 8   | 164K  | 3.46 | 50% |  8% |  2% | 0.020 |
| rank 32  | 557K  | 3.16 | 70% | 50% | 35% | 0.049 |
| rank 64  | 1.08M | 2.99 | 85% | 69% | 30% | 0.068 |
| **rank 128** | 2.13M | **2.88** | **87%** | 63% | **49%** | **0.094** |
| _depth ×2 (16 hops)_ | 328K | _3.54_ | _71%_ | _16%_ | _3%_ | _0.029_ |

Width is a clean lever: bpb falls monotonically (3.46 → 2.88), coverage rises across
thresholds, and causal *strength* (median Δlogits) climbs ~5×. At rank ≥ 64 the
probed sites are **100% load-bearing through hops 0–6** — only the final hop (a
saturation/refinement tail) is inert. v8 rank-8's plateau was **gradient starvation
from under-capacity** (grad-norm 0.04, fixed to 0.2–0.95 once widened), not the
architecture. **Depth** (more hops at fixed width) helped neither axis. The rank-128
model — fully readable, no embedding, 2.13M params — reaches 2.88 bpb, below the
opaque-leaning `v12` (3.13) and closing on the embedded `v13` baseline (2.26).

So the apparent "performance costs observability" tension was a *cross-architecture*
artifact (v8 vs v12); **within an architecture, adding the right capacity moves
performance and observability together** — a width-scaling law that is the first
solid evidence observability at no cost is reachable here. Caveat: these are still
near-bigram-capability models on a tiny corpus; the open question remains whether
it holds as the models get genuinely hard.

See [`apps/cli/observe.py`](../apps/cli/observe.py) for the tooling (`trace`,
`wordmap`, `causality`, `demo`, `sweep`, `coverage`).
