# v3: Associative Memory + Fourier Register Ops

**Status**: In progress. Not learning yet — Fourier projections too low-rank.

## Architecture
- Causal decay-weighted associative memory for cross-position mixing
- Parallel implementation: scores = Q·K^T, apply causal decay mask, retrieve = scores·V
- Per-step Fourier register ops for within-position transforms
- No attention, no embedding, no output projection

## Math
```
output_t = sum_{s<t} decay^(t-s-1) * (q_t · k_s) * v_s
```
Hopfield-style content-addressable memory. All 1970s math.

## Params
- ~328K total (8 steps, 128 channels, 16 Fourier basis)
- All projections via Fourier basis coefficients — very compact but low-rank

## Results (3xA40)
- 325K tok/s, 1.5s/step — fast
- Loss flat at ~6.93 (random) with 16 basis functions
- **Next**: try N_FOURIER_BASIS=128 for full-rank projections

## Run
```bash
TRAIN_BATCH_TOKENS=491520 GRAD_ACCUM_STEPS=16 N_FOURIER_BASIS=128 \
torchrun --standalone --nproc_per_node=3 train.py
```
