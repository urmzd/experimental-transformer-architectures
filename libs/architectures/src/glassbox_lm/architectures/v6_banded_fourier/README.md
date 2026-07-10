# v6_banded_fourier — Band-partitioned Fourier channel mixing with gated coupling

## Mechanism

Vocab-dim state (V = 1024). Input is one-hot. No output projection.

The fixed Fourier basis `Phi in R^{V x 2K}` is partitioned by harmonic
index into three bands — low / mid / high. Each band has its own learned
V -> C projection coefficients.

Per recurrent step:
- Cross-position:
  - Long-range memory: high-decay linear attention, projections built
    from the low-frequency band.
  - Short-range memory: low-decay linear attention, projections built
    from the high-frequency band.
  - Gated coupling: long-range output multiplicatively gates short-range
    output.
- Within-position:
  - Three parallel band projections (low / mid / high) feed three channel
    streams, recombined through learned scalar weights.
  - Gated coupling: low-band stream gates the high-band stream.

Caveat: the frequency framing is structural. Vocab ids are arbitrary
(BPE), so "low-frequency over vocab-id ordering" is a parameterization
choice, not a meaningful signal decomposition. Relative to a single
Fourier basis (v3), this gives several independent low-rank mixers with
gated combination, which is a more flexible parameterization.

## Status

~824K params, val_loss 5.66, still descending per benchmark run.

## Run

```bash
MODEL_VERSION=v6_banded_fourier \
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m glassbox_lm.training
```

### Hyperparameters

| Env Var | Default | Description |
|---|---|---|
| `NUM_STEPS` | 8 | Number of recurrent steps |
| `N_FOURIER_BASIS` | 16 | Total sin/cos basis pairs |
| `N_CHANNELS` | 128 | Channel dimension |
| `BAND_SPLIT` | `4,4,8` | Low, mid, high basis allocation |
| `SLOW_DECAY_INIT` | 4.0 | Long-range linear-attn decay logit |
| `FAST_DECAY_INIT` | 2.0 | Short-range linear-attn decay logit |
