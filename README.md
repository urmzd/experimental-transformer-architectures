<p align="center">
  <h1 align="center">Glassbox LM</h1>
  <p align="center">
    A testbed for <em>observable</em> language models: hidden dimension equals vocabulary size, so every register state is a readable distribution over words. Built to understand how the computation works — not to win a benchmark.
    <br /><br />
    <a href="#quick-start">Quick Start</a>
    &middot;
    <a href="https://github.com/urmzd/glassbox-lm/issues">Report Bug</a>
    &middot;
    <a href="#model-versions">Model Versions</a>
  </p>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/urmzd/glassbox-lm" alt="License"></a>
</p>

The problem these experiments attack: frontier language models are **billions of opaque parameters with no faithful way to see how they compute** or to link internal activity to meaning. The goal here is **observability at no performance cost** — architectures (and the tooling around them) where you can see *how the computation works* **without giving up the quality of an opaque model**. Observability that costs accuracy isn't the win; observability that's *free* is. (Whether you can drop embeddings was never the question — opacity at scale is.) For the precise problem — why readable *I/O* isn't it, and the honest catches — see [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md).

## What This Is

A testbed for **observable language models that don't trade away performance** — the aim is to *understand how they operate* **and** match what an opaque model achieves. The motivation is the opacity of modern LMs: billions of parameters, no faithful way to trace what the computation does or to link one internal representation to the next.

The approach keeps everything in **vocabulary space**: every architecture shares one mechanism — **hidden dimension = vocab size**, no learned embedding, no output projection — so the register state IS a distribution over words at every step. You can read each intermediate state as "which words are active and how strongly," and watch a prediction form across recurrent steps. Interpretability by construction, not post-hoc probing.

To be clear about the bar: this is **performance *and* observability — observability at no cost.** Not a leaderboard chase for its own sake, but a readable model has to *match* an opaque one (`v13_with_embedding`) to make the point: observability that costs accuracy proves nothing; observability that's free is the result worth having. See [Observing the computation](#observing-the-computation).

## Observing the computation

The point of word-space states is that you can read the computation directly. [`apps/cli/observe.py`](apps/cli/observe.py) (`observe <cmd>`):

| Command | What it shows |
|---|---|
| `observe trace` | top-k active words in the register state after each step, for a prompt — watch a prediction form |
| `observe wordmap` | for v8, the learned word→word interaction matrix `W = U@Vᵀ + diag(d)` ("which word activates which") |
| `observe causality` | perturb one vocab dimension mid-computation, measure how far the predicted next-word distribution moves — is the readable state *load-bearing* or decorative? |
| `observe demo` | controlled faithfulness check: train v8 on a planted bigram (`next[i] = perm[i]`), then verify the map recovers it |
| `observe sweep` | map which (step, word) sites are causally load-bearing vs decorative, across depth |
| `observe coverage` | the observability number: fraction of readable active-word sites that are causally load-bearing, with robustness across τ thresholds |
| `observe induction` | beyond-bigram check: train on in-context key→value pairs and test query-position recall against bigram chance *and* the key-free-copier ceiling |

**Faithfulness result (the central claim, tested).** On the planted-bigram demo, the recovered map `argmax_j W[i,j]` matches the true `perm[i]` for **100% of words** (chance ≈ 3%) — in this controlled setting, reading the weights tells the truth. The open questions are whether that readability holds on real text and at scale, and whether the states are causally load-bearing (run `observe causality` on a trained checkpoint to check). This is the actual research frontier — not the bits-per-byte score.

## What We've Found So Far

> **Precision note (methodology corrected 2026-06-09):** every table below predates a control-tensor fix. A stray `"weight"` entry in `CONTROL_TENSOR_NAME_PATTERNS` matched every `nn.Linear` weight, so all models except `v8_lowrank_vv` (custom `U`/`V` parameter names) kept fp32 master weights instead of the intended bf16. Forward and backward always ran under bf16 autocast, so the loss/bpb numbers stay comparable, but parameter memory, `raw_bytes`, and optimizer-state size were not uniform across models. Numbers are kept as published; re-run under the corrected regime before citing memory figures or close margins.

### Historical benchmark (10 min, 3× A40, batch 491,520 tokens)

| `MODEL_VERSION` | Architecture | Params | Steps | val_loss | val_bpb | tok/s | Status |
|---|---|---|---|---|---|---|---|
| **v8_lowrank_vv** (rank 8) | Recurrent rank-r V x V linear layer | **164K** | 100 | **5.24** | **3.10** | 270K | Still descending |
| v2_conv | Depthwise causal conv + Fourier channel mix | 353K | 464 | 5.39 | 3.19 | 383K | Still descending |
| v6_banded_fourier | Band-partitioned Fourier with gated coupling | 824K | 166 | 5.66 | 3.35 | 136K | Still descending |
| v1_shared_attn | Shared GQA attention + Fourier channel mix | 3.4M | 239 | 6.06 | 3.59 | 196K | Plateaued |
| v7_soft_ops | Soft op-bank + soft register addressing | 329K | 348 | 6.26 | 3.71 | 287K | Unstable (loss spikes) |
| v3_fourier_linattn | Linear attn with causal decay (Fourier proj) | 329K | 397 | 6.81 | 4.03 | 326K | Stuck |
| v8_lowrank_vv (rank 64) | Recurrent rank-r V x V linear layer | 1.1M | 188 | — | — | 270K | Memorized (train 0.04, overfitting) |

### Head-to-head on the target corpus (1× RTX 4090, identical 720s train budget, 2026-05-31)

Re-run on this project's actual target data (`willdepueoai/parameter-golf`, FineWeb sp1024) under a single budget-matched config (batch 491,520, 20 warmup steps, flat LR, grad-clip 1.0, 720s wall-clock cap):

| `MODEL_VERSION` | Params | Steps | val_loss | val_bpb |
|---|---|---|---|---|
| **v12_vocab_slice** | 4.20M | 589 | 5.383 | **3.188** |
| v14_data_dependent | 2.54M | 211 | 5.685 | 3.367 |
| v8_lowrank_vv (rank 8) | 0.16M | 229 | 5.984 | 3.544 |

On this corpus and an equal budget the ranking **inverts** the historical table: `v12_vocab_slice` wins on absolute bpb, while `v8_lowrank_vv` leads only on a *per-parameter* basis (25× smaller for +0.36 bpb). All three descended monotonically with no instability. Caveat: this is roughly **1/30th** of the Parameter Golf compute budget (1× RTX 4090 ≈ 12 min vs 8× H100 ≈ 10 min), so these are undertrained floors, not ceilings.

### Where this lands on Parameter Golf

Used here as a **raw-capability yardstick — not the project's goal** (the goal is interpretability/observability; see *What's actually unique here*). [OpenAI's Parameter Golf](https://github.com/openai/parameter-golf): the best LM in a **16 MB artifact** (code + compressed weights), trained in **under 10 min on 8× H100**, scored by **tokenizer-agnostic bits-per-byte** on the FineWeb validation set. `train.py` already emits the int8 + zlib artifact this competition scores.

| | bits-per-byte |
|---|---|
| Best verified leaderboard record (2026-05) | **~1.061** |
| Naive baseline (9-layer, 512-dim transformer) | **1.2244** |
| Published "best readable" claim (`v8_lowrank_vv` rank 128, 2026-06 width sweep) | 2.88 — under revision (methodology corrected 2026-06-09: matches no committed manifest; see [Width sweep](#width-sweep-2026-06-under-revision)) |
| Best readable result here (`v12_vocab_slice`, 2026-05 head-to-head) | **3.19** |

We are **~2.6× worse than even the naive baseline** — as-is, nothing here lands on the leaderboard. The prime suspect is the core constraint itself: the baseline is a conventional transformer with a 512-dim **embedding**, whereas the no-embedding `hidden_dim = vocab_size = 1024` design forces all computation through a vocab-space bottleneck. We tested this directly with the `v13_with_embedding` baseline (below) — and the embedding closes ~43% of the gap, confirming the constraint, not the compute budget, is the wall.

### The cost of readability, measured (3 GPUs, identical 600s train budget)

`v13_with_embedding` shares v12's exact body and differs only by a learned `Embedding(V, d) → Linear(d, V)`. From the committed run manifests (`artifacts/benchmark_results.json`, runs `0171d9fb` = `v13_embed` and `ae6c82a6` = `v12_sparse`, the pre-rename version ids):

| `MODEL_VERSION` | Embedding? | Params | Steps | val_bpb |
|---|---|---|---|---|
| **v13_with_embedding** | yes (baseline) | 4.46M | 603 | **2.368** |
| v12_vocab_slice | no | 4.20M | 636 | 3.236 |

Adding the embedding drops bpb **3.24 → 2.37 (≈0.87 bpb)** and closes ~43% of the gap to the 1.2244 baseline — in fewer steps.

> **Under revision (methodology corrected 2026-06-09: previously published values matched no committed manifest).** This table previously read v13 4.46M / 894 steps / 2.256 bpb, v12 4.20M / 945 steps / 3.079 bpb, and v8_lowrank_vv 0.16M / 477 steps / 3.427 bpb on "1× H100, 600 s", with final grad-norms 2.56 / 0.28 / 0.04 and a gradient-starvation reading of v8. No committed manifest records those runs, the committed logs contain no per-step `grad_norm` series, and the v8 row has no committed 600 s counterpart at all (the committed rank-8 sweep artifact is a 165-step, ~300 s run at 2.87 bpb). Treat the grad-norm narrative as unverified until the comparison is re-run.

**Framing:** the bar is **performance *and* observability — readability that costs nothing.** So the ~0.87 bpb gap between v12 (readable) and `v13_with_embedding` (opaque) is **not a price to accept — it is the gap to eliminate** while keeping the state readable. `v13` is the **performance target**, not a template to copy; the win is a readable model that reaches it. Closing this gap (via training recipe, capacity, and architecture) *without* losing observability is the core research program.

### Width sweep (2026-06): under revision

The hypothesis: scaling v8's interaction **rank** improves bits-per-byte and causal coverage (`observe coverage`: the fraction of readable active-word sites that are causally load-bearing) at the same time. The committed sweep artifacts (`artifacts/rank{8,16,32,64}_manifest.json`; 165 steps, ~300 s wall-clock, 3 GPUs per run) do not support it yet:

| v8 width | params | train loss | val_bpb |
|---|---|---|---|
| rank 8 | 164K | 4.85 | 2.87 |
| rank 16 | 295K | 5.39 | 3.19 |
| rank 32 | 557K | 4.35 | 2.56 |
| rank 64 | 1.08M | 0.87 | 0.47 |

bpb is not monotone in rank, and the rank-64 run is a memorization/leakage signal (0.47 bpb would beat the Parameter Golf record by more than 2×), echoing the rank-64 memorization in the historical run below. A previous version of this section claimed bpb falling monotonically 3.46 → 2.88 through rank 128, coverage rising 50% → 87%, and "no memorization at any rank"; those numbers match no committed manifest and no rank-128 artifact exists, so they are **under revision** (methodology corrected 2026-06-09) — full audit note in [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md). `scripts/rank_sweep.sh` regenerates the sweep with manifests at three seeds.

### What these results mean

> **Note:** the analysis below reflects the original 3× A40 run. The head-to-head and embedding-control results above supersede parts of it (on the target corpus, v12 > v8 at rank 8; the embedding control reaches 2.37 bpb). The width-sweep claims that previously superseded the rank-64 memorization finding are under revision; the committed rank-64 sweep artifact shows the same memorization signal.

**The low-rank V x V linear layer (v8) is the best architecture so far.** At rank 8 with 164K params, it reaches val_loss 5.24 in 100 steps — better than v2_conv (353K params, 464 steps) with half the parameters in one-fifth the steps. The train/val gap is essentially zero, confirming it's learning, not memorizing.

**But at rank 64, the same architecture memorizes.** The 1.1M-param version drove train_loss to 0.04 while val_loss stayed high. The rank-64 `U @ V^T` matrix has enough capacity to store a bigram lookup table. The committed 2026-06 sweep artifact points the same way: `artifacts/rank64_manifest.json` records train loss 0.87 / val 0.47 bpb — implausibly below the Parameter Golf record, a memorization/leakage signal rather than generalization. (An earlier revision of this README claimed the 2026-06 sweep found no memorization at any rank; that claim matched no committed artifact and is withdrawn.)

**This is still far from useful.** val_loss 5.24 (3.10 bpb) is well above the ~1.7 loss needed for 1 bpb. GPT-2 at 124M params achieves ~0.93 bpb. We're at 164K params, so the comparison isn't fair, but the gap is large.

### What's actually unique here

1. **hidden_dim = vocab_size with no embedding or output projection.** No published architecture does this. The state IS the prediction at every step.

2. **Interpretability by construction.** You can read intermediate states as distributions over vocab dimensions. This is not a post-hoc technique.

3. **The specific combination** of vocabulary-space state + various cross-position mechanisms (conv, decay memory, low-rank linear) + recurrent depth has not been explored before.

### What's NOT unique

- Weight sharing across depth: Universal Transformer (2019), ALBERT (2020), DEQ (2019) all do this.
- Fourier parameterization: FNet (2022), butterfly matrices, Fourier Neural Operators.
- Causal decay memory / linear attention: RWKV, Mamba, S4 all use equivalent mechanisms.
- Low-rank dimension-to-dimension interaction: mathematically, `x @ U @ V^T` is just a rank-r linear layer.
- Recurrent register machines: Neural Turing Machine (2014), Neural GPU (2016).

### Honest assessment of the v8_lowrank_vv results

The rank-8 variant works well because **direct bilinear dimension-to-dimension interaction is a good inductive bias when the dimensions are vocab entries**. Language is fundamentally about which words predict which other words. A model that directly parameterizes `W[i, j] = "dim i predicts dim j"` captures this structure more efficiently than architectures that must discover it through generic operations (convolutions, MLPs, Fourier transforms).

But this is a well-known insight. Bigram and n-gram models encode the same structure. The open question is whether multi-hop propagation (8 hops through the low-rank interaction matrix) captures longer-range dependencies that simple n-grams cannot. The current results don't answer this — we'd need to test on tasks requiring longer-range reasoning.

## Architecture

All variants share the same skeleton:

```
Input:  one-hot("cat") -> R["cat"] = 1.0, everything else 0.0
Repeat N times:
  1. Cross-position mixing  (how do words at different positions interact?)
  2. Within-position transform  (how do vocab-dim activations combine?)
Output: register state -> softcap -> cross-entropy loss
```

No embedding. No output projection. (Except `v13_with_embedding`, the opaque baseline kept for comparison.)

## Model Versions

Names describe mechanism, not metaphor.

### Core variants

| `MODEL_VERSION` | Cross-position | Within-position | Notes |
|---|---|---|---|
| `v1_shared_attn` | GQA + RoPE (weights shared across depth) | Fourier-parameterized channel mix | 3.4M params, plateaus early |
| `v2_conv` | Depthwise causal 1D conv | Fourier-parameterized channel mix | 353K params, strong baseline |
| `v3_fourier_linattn` | Linear attn with causal decay; Q/K/V/O via Fourier basis | Fourier-parameterized channel mix | Stuck — Fourier parameterization bottleneck |
| `v4_weight_shared` | Shared Q/K + per-head decay (v3 body) | Factored (diag + low-rank) channel mix | Size-reduction ablation of v3 |
| `v5_fft_linattn` | Linear attn with causal decay; Q/K/V/O via rFFT | FFT-based channel mix | Fourier-over-vocab same caveat |
| `v6_banded_fourier` | Band-partitioned Fourier linattn, gated coupling | Three parallel band projections, gated | 824K, still descending |
| `v7_soft_ops` | Linear attn with causal decay | Gumbel-soft op-bank + soft register addressing | Unstable, loss spikes |
| `v8_lowrank_vv` | Diagonal Q/K linear attn, activation similarity | Low-rank V x V (`U @ V^T + diag`) | **Best architecture** at rank 8 (historical run); 2026-06 width-sweep claims (rank 128 at 2.88 bpb) under revision — no committed manifest |
| `v9_linattn` | Linear attn with causal decay (dense projections) | MLP bottleneck | 4.2M params, best non-attention variant |
| `v10_state_cond_op` | Linear attn in compressed state space | State-conditioned soft read/op/write dispatch | Untested |
| `v11a_mixed_ops` | High-decay EMA + linear-attn | Sigmoid gate, dense layer, low-decay EMA | Trains; bf16 `arange` position bug fixed (2026-05) |
| `v11b_hard_routing` | Multi-timescale linear attn | Gumbel-hard op routing + PonderNet halting | Untested |
| `v12_vocab_slice` | Causal decay in fixed k-dim slice | MLP in k-dim slice | Best on target corpus (3.19 bpb, 2026-05); slice indices are deterministic vocab-id windows |
| `v14_data_dependent` | Input-modulated conv (Hyena) | Data-dependent decay (Mamba), DCT mix | Mamba / RWKV / Hyena bundle; bf16 dtype bug fixed (2026-05) |
| `v15_aux_loss` | v12 body + per-step CE + top-k sparsity | Entropy-adaptive write scaling | Training-side additions on v12 |
| `v16_multi_branch` | Per-column decay memory | Branched gated MLP + cross-column inhibition | Ensemble + gated branches |

### Opaque baseline (kept for comparison)

| `MODEL_VERSION` | Purpose |
|---|---|
| `v13_with_embedding` | **Opaque-embedding baseline** (for comparison only). Adds `Embedding(V, d) -> Linear(d, V)` before the register state (same body as `v12_vocab_slice`). Exists solely to measure what readability costs (≈0.87 bpb); the rest of the project keeps the state readable. Not a template. |

## Quick Start

```bash
# Setup on RunPod
curl -sSL https://raw.githubusercontent.com/urmzd/glassbox-lm/main/setup.sh | bash

# Or manually. torch is a base dependency; on GPU images that already ship
# torch (e.g. RunPod), the preinstalled build satisfies it and is left alone.
uv pip install --system -e .
python data/download_data.py --variant sp1024

# Train the best model (low-rank V x V, rank 8)
INTERACTION_RANK=8 MODEL_VERSION=v8_lowrank_vv \
  torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) train.py

# Benchmark all models
benchmark

# Benchmark specific models
benchmark --versions v8_lowrank_vv,v2_conv,v14_data_dependent --minutes 10
```

All hyperparameters configurable via environment variables. See `core/config.py`.

## What We've Learned

**Inductive bias matters more than parameter count.** `v8_lowrank_vv` (164K params, rank 8) beats `v1_shared_attn` (3.4M params, 20x more) because direct dimension-to-dimension interaction is a better prior for language than generic attention in vocab space.

**Too much capacity in the right place enables memorization.** `v8_lowrank_vv` at rank 64 memorized the training batch in the original 3× A40 run (train loss 0.04) while rank 8 generalized, and the committed 2026-06 sweep artifact reproduces the signal (`artifacts/rank64_manifest.json`: 0.47 val bpb, implausibly below the Parameter Golf record). An earlier revision claimed the 2026-06 sweep showed train ≈ val at every rank through 128 and re-read the rank-8 result as gradient starvation; that claim matched no committed artifact and is under revision (methodology corrected 2026-06-09; see [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md)).

**Fourier-over-vocab parameterization is a structural bottleneck.** `v3_fourier_linattn` and `v5_fft_linattn` both constrain their linear-attention projections to linear combinations of sin/cos over vocab indices. Both got stuck. Vocab ids from BPE have no meaningful ordering, so "smooth over vocab ids" throws away useful capacity. The linear-attention core itself works fine — see `v9_linattn`, which uses dense projections on the same core.

**Attention in vocab space is expensive and unhelpful at this scale.** `v1_shared_attn` spends most of its 3.4M params on Q/K/V/O projections over V=1024 vectors and still plateaus at val_loss ~6.06. The overhead isn't justified.

**Training instability is a real problem.** `v7_soft_ops` had two catastrophic loss spikes (9.35 at step 161, 8.28 at step 181) before recovering. The soft op-selection path is fragile.

## Inspirations

- [Linear Genetic Programming](https://github.com/urmzd/linear-gp) — register machines, sequential cheap operations
- [Tangled Program Graphs](https://web.cs.dal.ca/~mheywood/) — hard bidding, multi-timescale memory
- Neural GPU (Kaiser 2016) — repeated convolution learns algorithms
- Deep Equilibrium Models (Bai 2019) — weight-shared iteration to convergence
- Mamba (Gu & Dao 2023) — data-dependent state transitions
- RWKV — linear attention with causal decay
- Hyena — input-dependent long convolutions

## Documentation

The docs map lives at [docs/README.md](docs/README.md). Key entries:

| Document | Question it answers |
|---|---|
| [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) | Why — the mechanism-vs-boundary framing of the observability problem |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How — repo layout, shared infrastructure, the model-variant contract, entry points |
| [docs/TESTING.md](docs/TESTING.md) | How it's verified — test gates, GPU-verified concerns, results discipline |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Working on the repo — setup, dev loop, add-a-model checklist |
| [AGENTS.md](AGENTS.md) | Operational reference — env vars, training commands, RunPod |

## Agent Skill

This repo's conventions are available as portable agent skills in [`skills/`](skills/).

## License

[Apache-2.0](LICENSE)
