# v3_fourier_linattn — Linear attention with causal decay, projections via Fourier basis

## Mechanism

Vocab-dim state (V = 1024). Input is one-hot. No output projection.

Per recurrent step:
- Cross-position: linear attention with exponential causal decay.
  ```
  output_t = sum_{s < t} decay^(t-s-1) * (q_t . k_s) * v_s
  ```
  Q/K/V/O are each V x C matrices, each built as `basis @ coeffs.T` from
  a fixed sin/cos basis over vocab indices and learned C x 2K
  coefficients. This constrains projections to a 2K-dim smooth-function
  subspace over vocab-id ordering.
- Within-position: Fourier-parameterized channel mixer (same structure).

## Status

Stuck. val_loss ~6.81. The Fourier parameterization is the bottleneck:
vocab ids are arbitrary (BPE), so constraining projections to "smooth
functions over vocab-id ordering" throws away most of the capacity that
would be useful. The linear-attention-with-decay core itself is sound
(same family as RWKV / Linear Transformer); swap the parameterization
for a dense projection and see v9_linattn for the corresponding working
variant.

## Params

~329K (8 steps, 128 channels, 16 Fourier basis).

## Run

```bash
MODEL_VERSION=v3_fourier_linattn \
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m glassbox_lm.training
```
