# v9_linattn — Linear attention with causal decay, dense projections

## Mechanism

Vocab-dim state (V = 1024). Input is one-hot. No output projection.

Per recurrent step:
- Cross-position: linear attention with exponential causal decay.
  ```
  output_t = sum_{s < t} decay^(t-s-1) * (q_t . k_s) * v_s
  ```
  Q / K / V / O are **dense learned** V -> r projections (unconstrained
  matrices). Each step has its own projection weights.
- Within-position: MLP bottleneck (down -> gelu -> up).

Equivalent formulation: maintain a running outer-product state matrix
`S_t = sum decay * k_s outer v_s`, updated once per position, queried as
`q_t @ S_t`. Same family as RWKV / Linear Transformer.

## Status

Best non-attention variant on the benchmark. val_bpb ~3.26 at step 228.
~4.2M params, ~199K tok/s.

## Contrast with v3 / v5

v3_fourier_linattn and v5_fft_linattn implement the same linear-
attention core, but constrain the V -> C projections to a Fourier (or
rFFT) parameterization. v9 uses unconstrained dense projections, which
is what fixes the stuck behavior observed in v3.

## Run

```bash
MODEL_VERSION=v9_linattn \
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m glassbox_lm.training
```

## Env vars

| Variable | Default | Notes |
|---|---|---|
| `STATE_DIM` | 64 | Linear-attention feature dim r |
| `INNER_DIM` | 128 | MLP bottleneck width |
| `NUM_STEPS` | 8 | Number of recurrent steps |
