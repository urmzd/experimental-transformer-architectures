# v2_conv — Depthwise causal conv + per-step Fourier channel mix

## Mechanism

Vocab-dim state (V = 1024). Input is one-hot. No output projection.

Per recurrent step (each step has its own independent weights):
- Cross-position: depthwise causal 1D conv with a small kernel. One
  filter per V dimension, left-padded for causality. Captures local
  context along the sequence axis with no cross-dimension mixing.
- Within-position: Fourier-parameterized channel mixer (two V x C
  projections built from a fixed sin/cos basis over vocab indices, with
  a C x C channel mix plus nonlinearity in between).

Caveat: the Fourier basis imposes a smoothness prior over vocab-id
ordering, which is arbitrary (BPE). Functionally a low-rank V -> C -> V
mixer with a non-generic parameterization.

## Status

353K params, val_loss 5.39 at step 464, still descending. Highest
throughput of any variant in the benchmark (~383K tok/s). Strong
baseline; surpassed only by v8_lowrank_vv.

## Run

```bash
MODEL_VERSION=v2_conv \
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m glassbox_lm.training
```
