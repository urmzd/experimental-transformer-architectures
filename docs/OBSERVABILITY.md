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
measures performance and observability *jointly*, and treats the ~0.87 bpb gap to
the opaque `v13` baseline as a gap to *eliminate*, not a price to pay.

One more limit worth stating plainly: this is a way to *build* observable models,
not to crack open an existing opaque one. It cannot be applied post-hoc to GPT-4 —
the architecture has to be designed this way. The bet is that capable models can
be observable *from the start*.

## Width sweep (2026-06): what the committed evidence supports

The hypothesis worth testing: scaling v8's interaction **width** (the rank of the
low-rank V×V layer) improves both axes at once — bits-per-byte for performance,
`observe coverage` for observability (the fraction of readable active-word sites
that are causally load-bearing). The committed sweep artifacts
(`artifacts/rank{8,16,32,64}_manifest.json`; model_version `v8_graph`, the
pre-rename id for `v8_lowrank_vv`; 165 steps, ~300 s wall-clock, 3 GPUs,
batch 491,520 tokens per run) record:

| v8 width | params | train loss | val loss | val bpb |
|---|---|---|---|---|
| rank 8  | 164K  | 4.85 | 4.84 | 2.87 |
| rank 16 | 295K  | 5.39 | 5.38 | 3.19 |
| rank 32 | 557K  | 4.35 | 4.33 | 2.56 |
| rank 64 | 1.08M | 0.87 | 0.79 | 0.47 |

Two things follow from the committed data. bpb is **not monotone in rank**
(2.87 → 3.19 → 2.56), and the rank-64 run is a **memorization/leakage signal,
not a result**: 0.47 val bpb at 1.08M params would beat the Parameter Golf
record (~1.061) by more than 2×, which is not credible. It points the same way
as the rank-64 memorization seen in the earlier 3× A40 run, not against it.

> **Under revision (methodology corrected 2026-06-09: the published sweep table
> matched no committed manifest).** This section previously claimed bpb 3.46
> (rank 8), 3.16 (rank 32), 2.99 (rank 64), 2.88 (rank 128); coverage rising
> 50% → 87% @τ=0.02; median Δlogits 0.020 → 0.094; a depth-×2 control at 3.54;
> "600 s each on 1× H100"; and "train ≈ val throughout — no memorization at any
> rank". None of that is reproducible from the committed artifacts: every
> recorded bpb differs, the runs were 3-GPU and ~300 s, no rank-128 manifest
> exists, and the rank-64 artifact contradicts the no-memorization claim
> outright. The coverage column has no committed source either. The old numbers
> are kept here for the record only; treat the "width-scaling law" and the
> "rank-8 plateau was gradient starvation" diagnosis as unverified hypotheses
> until `scripts/rank_sweep.sh` regenerates the sweep with committed manifests
> at multiple seeds.

> **Precision note (methodology corrected 2026-06-09):** v8's custom `U`/`V`
> parameter names always escaped the control-tensor bug, so the sweep rows above
> trained bf16. The cross-architecture references (`v12` at 3.24, `v13` at 2.37
> in `artifacts/benchmark_results.json`) come from runs whose master weights
> stayed fp32 because of a stray `"weight"` control pattern; forwards still ran
> bf16 autocast, so bpb is comparable, but the precision/memory regimes were not
> identical. Re-run the v8/v12/v13 comparison under the corrected regime before
> citing close margins.

See [`apps/cli/observe.py`](../apps/cli/observe.py) for the tooling (`trace`,
`wordmap`, `causality`, `demo`, `sweep`, `coverage`, `induction`).
