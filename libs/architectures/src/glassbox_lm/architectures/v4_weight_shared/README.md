# v4_weight_shared — Parameter-reduced variant of v3_fourier_linattn

## Mechanism

Same linear-attention-with-decay core as v3_fourier_linattn, with
aggressive weight sharing and factorization to land near ~101K params
(down from v3's 329K):

- Channel mix factored as diagonal + low-rank instead of dense C x C.
- Q and K projections shared across all steps; V and O remain per-step.
- Per-head decay (H=4) giving multiple effective timescales at minimal
  extra cost.
- Five unique step modules invoked twice each, with small per-invocation
  override scalars.
- Q/K normalization for training stability.

## Status

Not a new mechanism — size-reduction ablation of v3. Trained but not
competitive with v2_conv or v8_lowrank_vv.

## Run

```bash
MODEL_VERSION=v4_weight_shared \
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m glassbox_lm.training
```
