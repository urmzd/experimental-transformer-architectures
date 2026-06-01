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
load-bearing), scaling v8's interaction **width** improved *both at once*:

| v8 variant | params | bpb ↓ | grad-norm | coverage @τ=.05 ↑ |
|---|---|---|---|---|
| rank 8  | 164K | 3.46 | 0.04 (starved) |  8% |
| **rank 32** | 557K | **3.16** | **0.95** | **50%** |
| depth ×2 (16 hops) | 328K | 3.54 | 0.09 (starved) | 16% |

Going rank 8 → 32 cut bpb **and** raised coverage at every threshold (mean Δlogits
4.6×), fixed the gradient starvation, and did not memorize (train ≈ val). It also
beats the opaque-leaning `v12` on coverage at near-equal bpb with ~1/7 the params.
*Depth* helped neither axis. So the apparent "performance costs observability"
tension was a cross-architecture artifact; **within an architecture, adding the
right capacity moved performance and observability together** — the first concrete
sign that observability at no cost is reachable in this regime. Caveat: these are
still near-bigram-capability models; the open question remains whether this holds
as the models get genuinely hard.

See [`apps/cli/observe.py`](../apps/cli/observe.py) for the tooling (`trace`,
`wordmap`, `causality`, `demo`, `sweep`, `coverage`).
