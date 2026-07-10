# v1_shared_attn — Shared GQA attention + per-step Fourier channel mix

## Mechanism

Vocab-dim state (V = 1024). Input is one-hot. No output projection.

Per recurrent step:
- Cross-position: GQA self-attention with RoPE, **weights shared across all steps**.
- Within-position: independent per-step Fourier-parameterized channel mixer.
  Two V x C projections built as `basis @ coeffs.T` where `basis` is a
  fixed sin/cos grid over vocab indices. Channels are mixed by a dense
  C x C matrix plus nonlinearity.

The Fourier basis imposes a smoothness prior over vocab-id ordering. Vocab
ids are arbitrary (BPE), so this is a parameter-efficient low-rank
parameterization rather than a frequency decomposition of language.

## Status

Plateaus at val_loss ~6.06. Attention in vocab space is dominated by Q/K/V
projection cost (3M+ params) and the benefit does not justify the
overhead at this scale.

## Params

~3.4M total, most in the shared Q/K/V/O matrices.

## Run

```bash
MODEL_VERSION=v1_shared_attn \
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m glassbox_lm.training
```
