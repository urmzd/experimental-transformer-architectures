# TODO

## Training runs needed
- [ ] `v11b_hard_routing` — hard Gumbel routing, multi-timescale linattn, adaptive halting
- [x] `v11a_mixed_ops` — trained on GPU; descends, stable (bf16 `arange` position bug fixed 2026-05)
- [x] `v12_vocab_slice` — trained on GPU; best on target corpus (3.19 bpb, 2026-05)
- [x] `v14_data_dependent` — trained on GPU (bf16 dtype bug fixed 2026-05)
- [ ] `v10_state_cond_op` — state-conditioned soft read/op/write dispatch
- [ ] `v7_soft_ops` — differentiable register machine (address instability)
- [ ] `v8_lowrank_vv` — extended run at rank 8, measure long-range behavior (720s budget run done; long-range probe still needed)

## Infrastructure
- [x] Per-step gradient-norm logging (`train.py` logs `grad_norm`; manifest stores `final_grad_norm`)
- [ ] MLX support for current models — only v0 has an MLX training script
- [ ] Wandb/tensorboard logging
- [ ] Add `v11b_hard_routing`, `v11a_mixed_ops`, `v12_vocab_slice` to `run_all.py` model list

## Training
- [x] Checkpoint save/resume
- [x] Roundtrip eval optional (ROUNDTRIP_EVAL=1)
- [ ] Learning rate warmup schedule (currently flat after warmup steps)
- [ ] Gumbel temperature annealing for `v11b_hard_routing` — anneal tau from 1.0 → 0.1 during training

## The objective: performance *and* observability (observability at no cost)

The thesis is that you can have **both** — a model whose computation is readable *by construction* (every hidden state is a distribution over words) that **also matches an opaque model's performance**. Success is not "readable but worse"; it is **readable *and* as good**. Two fronts, pursued together:

### A. Observability — is the readable state real?

Tooling: `apps/cli/observe.py` (`observe trace|wordmap|causality|demo`).

- [x] Observability trace (`observe trace`): top-k active vocab dims of the register state at each step
- [x] `v8_lowrank_vv` word→word map (`observe wordmap`); **planted-bigram faithfulness check passes at 100% recovery** (`observe demo`, chance ≈3%)
- [~] Causality probe built (`observe causality`); needs a **trained checkpoint** to return a real load-bearing/decorative verdict (untrained models read as decorative, correctly)
- [ ] Faithfulness/causality on **real text + a trained checkpoint** (the prior runs didn't save weights — retrain and `save_checkpoint`, then probe)
- [ ] Define a quantitative interpretability metric beyond planted recovery (e.g. state→output agreement, human-legible rationale) and measure across variants
- [ ] Track observability *jointly with* bpb as capability improves — does the state stay readable as the model gets better? (the entire claim is that it can)

### B. Performance — close the gap *without* losing observability

A readable model must reach the opaque baseline. Measured against [Parameter Golf](https://github.com/openai/parameter-golf) (16 MB artifact, <10 min on 8× H100, bits-per-byte on FineWeb).
Current standing: readable variants ~3.1–3.4 bpb; opaque `v13` **2.26**; naive baseline 1.2244; best record ~1.061. **The job: close the ~0.82 bpb readable-vs-opaque gap (and beyond) while keeping the state legible.** (See README.)

- [x] **Decisive experiment (done 2026-05-31, 1× H100, 600s each):** `v13_with_embedding` **2.26 bpb** vs `v12_vocab_slice` (no-embed) 3.08 vs `v8_lowrank_vv` 3.43. The embedding closes ~45% of the gap to the 1.2244 baseline (~0.82 bpb). **The no-embedding constraint, not compute, is the wall** — so the program is to *break that wall while keeping the state readable*, not to accept it. (A full 8× H100 / 8B-token run would lower absolute numbers but not change the relative finding.)
- [ ] **Training recipe (closes the bpb gap; verified from winning entries):** Polar-Express Muon optimizer + warmdown/MIN_LR floor; fused LeakyReLU(0.5)² + softcapped-CE Triton kernels (winners hit ~4900 steps/600s); depth recurrence + parallel residual lanes (effective depth at ~zero param cost)
- [ ] **Score-first "legal" test-time training at eval** — highest single verified ROI (−0.0337 bpb in one PR); must honor causality / score-before-update / single-pass / no pre-quant TTT on val
- [ ] **Compression to fit more params under 16 MB (verified):** GPTQ int6/int7 + LQER rank-4 int4 correction + SDClip std-based clipping + L1 similarity-sort + lrzip-zpaq/brotli
- [ ] Sliding-window stride-64 evaluation (Parameter Golf's eval method; ~0.01 bpb vs fixed-context)
- [ ] **Pilot TurboQuant** as an artifact-size lever (rotation + per-coordinate Lloyd-Max + rANS; data-oblivious, zero-calibration → no training-time cost) — validate at hidden_dim=1024, where the coordinate-independence assumption is weaker than the paper's high-dim regime (arXiv 2504.19874)
- [x] ~~Gemma 4 "fast sampling"~~ — investigated: it is MTP *speculative decoding* (inference-only, distribution-preserving) → cannot move bpb and would only burn the 16 MB budget on a 2nd network. Not applicable.
- Note: winners lean on tied embeddings + SP8192 vocab — levers the no-embedding `hidden=vocab=1024` design cannot use directly; re-tune layer counts/dims
