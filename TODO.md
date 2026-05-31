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

## Parameter Golf (the actual objective)

Target: best LM in a 16 MB artifact, <10 min on 8× H100, scored by bits-per-byte on FineWeb.
Current standing: ~3.1–3.6 bpb vs the 1.2244 verified naive baseline / ~1.061 verified best record — ~3× off. We are losing on training *fundamentals* (optimizer, step count, recipe), not just compression. (See README.)

- [ ] **Decisive experiment:** full 8× H100 / 10-min / 8B-token run of `v12_vocab_slice` + `v8_lowrank_vv` + `v13_with_embedding` (control) — separates the no-embedding constraint's cost from the compute-budget shortfall
- [ ] **Training recipe (closes the bpb gap; verified from winning entries):** Polar-Express Muon optimizer + warmdown/MIN_LR floor; fused LeakyReLU(0.5)² + softcapped-CE Triton kernels (winners hit ~4900 steps/600s); depth recurrence + parallel residual lanes (effective depth at ~zero param cost)
- [ ] **Score-first "legal" test-time training at eval** — highest single verified ROI (−0.0337 bpb in one PR); must honor causality / score-before-update / single-pass / no pre-quant TTT on val
- [ ] **Compression to fit more params under 16 MB (verified):** GPTQ int6/int7 + LQER rank-4 int4 correction + SDClip std-based clipping + L1 similarity-sort + lrzip-zpaq/brotli
- [ ] Sliding-window stride-64 evaluation (Parameter Golf's eval method; ~0.01 bpb vs fixed-context)
- [ ] **Pilot TurboQuant** as an artifact-size lever (rotation + per-coordinate Lloyd-Max + rANS; data-oblivious, zero-calibration → no training-time cost) — validate at hidden_dim=1024, where the coordinate-independence assumption is weaker than the paper's high-dim regime (arXiv 2504.19874)
- [x] ~~Gemma 4 "fast sampling"~~ — investigated: it is MTP *speculative decoding* (inference-only, distribution-preserving) → cannot move bpb and would only burn the 16 MB budget on a 2nd network. Not applicable.
- Note: winners lean on tied embeddings + SP8192 vocab — levers the no-embedding `hidden=vocab=1024` design cannot use directly; re-tune layer counts/dims
