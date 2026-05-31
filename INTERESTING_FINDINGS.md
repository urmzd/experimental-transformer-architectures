# Interesting Findings

> **Caveat (2026-05):** The table below is a *synthetic-data, CPU forward/backward* micro-benchmark — it measures speed and init-time gradient/loss behavior, **not trained quality**. OpenAI's Parameter Golf write-up found that synthetic benchmarks don't transfer to real-text bits-per-byte, so don't use this to rank architectures for the real objective. For trained quality on the target corpus, see the head-to-head table in [`README.md`](README.md). (The `--iters/--batch/--seq-len` invocation below predates the current `apps/cli/benchmark.py`, which now takes `--versions/--minutes/--batch`.)

## Benchmark: CPU Forward+Backward (synthetic data, no training)

Run via `python apps/cli/benchmark.py --iters 3 --batch 2 --seq-len 128`.

| Model | Params | Raw MB | ~Int8 MB | Fwd ms | Bwd ms | Total ms | tok/s | Init Loss | AvgGrad | MaxGrad | Dead |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v5_fft_linattn | 919K | 3.67 | 0.92 | 34 | 57 | 91 | 7556 | 7.52 | 2.02e+00 | 1.99e+01 | 0 |
| v3_fourier_linattn | 329K | 1.31 | 0.33 | 648 | 572 | 1221 | 395 | 7.54 | 1.85e+00 | 1.58e+01 | 0 |
| v7_soft_ops | 132K | 0.53 | 0.13 | 158 | 218 | 376 | 1618 | 7.59 | 2.04e+00 | 2.51e+01 | 0 |
| v8_lowrank_vv | 1081K | 4.33 | 1.08 | 215 | 327 | 541 | 1193 | 13.55 | 2.86e+00 | 1.00e+02 | 0 |
| v4_weight_shared | 102K | 0.41 | 0.10 | 801 | 708 | 1509 | 319 | 21.57 | 5.60e+00 | 5.10e+01 | 0 |
| v9_linattn | 4195K | 16.78 | 4.20 | 16 | 30 | 46 | 15716 | 23.15 | 3.79e-01 | 1.25e+01 | 0 |
| v2_conv | 353K | 1.41 | 0.35 | 627 | 1105 | 1732 | 408 | 23.40 | 2.08e-01 | 1.22e+01 | 0 |
| v1_shared_attn | 3360K | 13.44 | 3.36 | 263 | 4729 | 4992 | 973 | 23.44 | 3.21e-01 | 1.20e+01 | 5 |
| v10_state_cond_op | 1387K | 5.55 | 1.39 | 15 | 25 | 40 | 17026 | 23.63 | 1.15e-01 | 1.21e+01 | 0 |
| v6_banded_fourier | 824K | 3.30 | 0.82 | 1471 | 1378 | 2848 | 174 | 23.63 | 4.30e-02 | 1.21e+01 | 0 |

### Observations

**Dense projections are 30-70x faster than Fourier-parameterized models on CPU.** `v9_linattn` (46ms) and `v10_state_cond_op` (40ms) use dense `nn.Linear` projections and are dramatically faster than Fourier-parameterized variants like `v3_fourier_linattn` (1221ms), `v4_weight_shared` (1509ms), and `v6_banded_fourier` (2848ms) — despite having more parameters. The Fourier-basis matmuls and softmax-over-vocab operations are expensive when hidden_dim = vocab_size.

**Fourier-parameterized models have better initialization loss.** `v3_fourier_linattn` (7.54), `v5_fft_linattn` (7.52), and `v7_soft_ops` (7.59) start with much lower loss on random data than dense-projection models (`v9_linattn` 23.15, `v10_state_cond_op` 23.63). The structured basis gives the model a head start at init. Whether this translates to better trained performance is a separate question — it didn't for v3 / v5 / v6, where the same parameterization that helps at init becomes a capacity bottleneck under training.

**`v5_fft_linattn` is the speed/quality sweet spot among Fourier variants.** Best init loss (7.52) while being 13x faster than `v3_fourier_linattn` and 4x faster than `v7_soft_ops`. The rFFT-based projection avoids the explicit basis matmul overhead.

**`v1_shared_attn` has 5 dead parameters (zero gradients).** May indicate unused capacity or initialization issues in the shared attention block.

**`v4_weight_shared` is the smallest (102K params) but the second slowest.** Factored channel mix and step-reuse loops add CPU overhead beyond what the parameter reduction saves.

**`v8_lowrank_vv` has the highest max gradient (100).** Suggests potential training instability — the recurrent V x V interaction may need gradient clipping or a careful LR schedule. Matches the training observations (at rank 64 it memorizes; at rank 8 it generalizes).

**`v6_banded_fourier` is the slowest model by far (2848ms).** Three independent band projections with cross-band gating compound the cost of Fourier operations.
