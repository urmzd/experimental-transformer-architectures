<p align="center">
  <h1 align="center">Experimental Transformer Architectures</h1>
  <p align="center">
    A testbed for <em>observable</em> language models: hidden dimension equals vocabulary size, so every register state is a readable distribution over words. Built to understand how the computation works — not to win a benchmark.
    <br /><br />
    <a href="#quick-start">Quick Start</a>
    &middot;
    <a href="https://github.com/urmzd/experimental-transformer-architectures/issues">Report Bug</a>
    &middot;
    <a href="#model-versions">Model Versions</a>
  </p>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/urmzd/experimental-transformer-architectures" alt="License"></a>
</p>

The problem these experiments attack: frontier language models are **billions of opaque parameters with no faithful way to see how they compute** or to link internal activity to meaning. The goal here is **understanding how the system works and operates** — architectures (and the tooling around them) where the computation is observable *by construction*. Whether you can drop embeddings is not the question; embeddings were never the problem — opacity at scale is.

## What This Is

A testbed for **observable language models** — built so you can understand *how they operate*, not just measure how well they score. The motivation is the opacity of modern LMs: billions of parameters, no faithful way to trace what the computation does or to link one internal representation to the next.

The approach keeps everything in **vocabulary space**: every architecture shares one mechanism — **hidden dimension = vocab size**, no learned embedding, no output projection — so the register state IS a distribution over words at every step. You can read each intermediate state as "which words are active and how strongly," and watch a prediction form across recurrent steps. Interpretability by construction, not post-hoc probing.

To be clear about what this is *not*: it is not an attempt to drop embeddings or to beat a leaderboard. The small parameter counts and the no-embedding design are **means to observability**. One variant (`v13_with_embedding`) deliberately reintroduces an opaque embedding as a labeled control — a way to measure what observability costs (≈0.82 bpb; see below), not a template to copy.

## Observing the computation

The point of word-space states is that you can read the computation directly. [`apps/cli/observe.py`](apps/cli/observe.py) (`observe <cmd>`):

| Command | What it shows |
|---|---|
| `observe trace` | top-k active words in the register state after each step, for a prompt — watch a prediction form |
| `observe wordmap` | for v8, the learned word→word interaction matrix `W = U@Vᵀ + diag(d)` ("which word activates which") |
| `observe causality` | perturb one vocab dimension mid-computation, measure how far the predicted next-word distribution moves — is the readable state *load-bearing* or decorative? |
| `observe demo` | controlled faithfulness check: train v8 on a planted bigram (`next[i] = perm[i]`), then verify the map recovers it |

**Faithfulness result (the central claim, tested).** On the planted-bigram demo, the recovered map `argmax_j W[i,j]` matches the true `perm[i]` for **100% of words** (chance ≈ 3%) — in this controlled setting, reading the weights tells the truth. The open questions are whether that readability holds on real text and at scale, and whether the states are causally load-bearing (run `observe causality` on a trained checkpoint to check). This is the actual research frontier — not the bits-per-byte score.

## What We've Found So Far

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
| Best here (`v12_vocab_slice`) | **3.19** |

We are **~2.5–3× worse than even the naive baseline** — as-is, nothing here lands on the leaderboard. The prime suspect is the core constraint itself: the baseline is a conventional transformer with a 512-dim **embedding**, whereas the no-embedding `hidden_dim = vocab_size = 1024` design forces all computation through a vocab-space bottleneck. We tested this directly with the `v13_with_embedding` control (below) — and the embedding closes ~45% of the gap, confirming the constraint, not the compute budget, is the wall.

### The no-embedding tax, measured (1× H100, identical 600s train budget, 2026-05-31)

`v13_with_embedding` shares v12's exact body and differs only by a learned `Embedding(V, d) → Linear(d, V)`:

| `MODEL_VERSION` | Embedding? | Params | Steps | val_bpb | final grad-norm |
|---|---|---|---|---|---|
| **v13_with_embedding** | yes (control) | 4.46M | 894 | **2.256** | 2.56 |
| v12_vocab_slice | no | 4.20M | 945 | 3.079 | 0.28 |
| v8_lowrank_vv | no | 0.16M | 477 | 3.427 | 0.04 |

Adding the embedding drops bpb **3.08 → 2.26 (≈0.82 bpb)** and closes ~45% of the gap to the 1.2244 baseline — in fewer steps. The per-step gradient norm (logged via `grad_norm`) tracks this exactly: v8 is gradient-starved (~0.04 — clipping never fires, it plateaus), v12 is healthy (~0.28), v13 learns hard (~2.56 — gradient clipping engages). v8's 164K capacity is an additional wall: it stops improving with more compute, while v12 keeps descending (3.19 → 3.08 with more steps).

**Framing:** the readable vocab-space state is the *point* of this project — **interpretability and observability**, not the leaderboard. This "tax" measures what readability costs in raw modeling capability; whether the readable states deliver understanding worth ~0.8 bpb is the real open question, and one these bits-per-byte benchmarks do not measure. `v13_with_embedding` is the deliberately-opaque control here, not a template to copy.

### What these results mean

> **Note:** the analysis below reflects the original 3× A40 run and is **superseded** by the head-to-head and no-embedding-tax results above (on the target corpus, v12 > v8; the embedding control reaches 2.26 bpb).

**The low-rank V x V linear layer (v8) is the best architecture so far.** At rank 8 with 164K params, it reaches val_loss 5.24 in 100 steps — better than v2_conv (353K params, 464 steps) with half the parameters in one-fifth the steps. The train/val gap is essentially zero, confirming it's learning, not memorizing.

**But at rank 64, the same architecture memorizes.** The 1.1M-param version drove train_loss to 0.04 while val_loss stayed high. The rank-64 `U @ V^T` matrix has enough capacity to store a bigram lookup table. Rank 8 can't, so it's forced to learn a compressed, generalizable mapping instead.

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

No embedding. No output projection. (Except `v13_with_embedding`, the labeled control.)

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
| `v8_lowrank_vv` | Diagonal Q/K linear attn, activation similarity | Low-rank V x V (`U @ V^T + diag`) | **Best so far at rank 8** |
| `v9_linattn` | Linear attn with causal decay (dense projections) | MLP bottleneck | 4.2M params, best non-attention variant |
| `v10_state_cond_op` | Linear attn in compressed state space | State-conditioned soft read/op/write dispatch | Untested |
| `v11a_mixed_ops` | High-decay EMA + linear-attn | Sigmoid gate, dense layer, low-decay EMA | Trains; bf16 `arange` position bug fixed (2026-05) |
| `v11b_hard_routing` | Multi-timescale linear attn | Gumbel-hard op routing + PonderNet halting | Untested |
| `v12_vocab_slice` | Causal decay in fixed k-dim slice | MLP in k-dim slice | Best on target corpus (3.19 bpb, 2026-05); slice indices are deterministic vocab-id windows |
| `v14_data_dependent` | Input-modulated conv (Hyena) | Data-dependent decay (Mamba), DCT mix | Mamba / RWKV / Hyena bundle; bf16 dtype bug fixed (2026-05) |
| `v15_aux_loss` | v12 body + per-step CE + top-k sparsity | Entropy-adaptive write scaling | Training-side additions on v12 |
| `v16_multi_branch` | Per-column decay memory | Branched gated MLP + cross-column inhibition | Ensemble + gated branches |

### Control variant

| `MODEL_VERSION` | Purpose |
|---|---|
| `v13_with_embedding` | **Thesis-breaking control.** Adds `Embedding(V, d) -> Linear(d, V)` before the register state (same body as `v12_vocab_slice`). Exists to measure what the no-embedding constraint costs; do not reuse as a template. |

## Quick Start

```bash
# Setup on RunPod
curl -sSL https://raw.githubusercontent.com/urmzd/experimental-transformer-architectures/main/setup.sh | bash

# Or manually
uv pip install --system -r pyproject.toml
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

**Too much capacity in the right place enables memorization.** `v8_lowrank_vv` at rank 64 memorizes the training batch (train loss 0.04). At rank 8 it generalizes (train ≈ val). The constraint forces learning.

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

## Agent Skill

This repo's conventions are available as portable agent skills in [`skills/`](skills/).

## License

[Apache-2.0](LICENSE)
