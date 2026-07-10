# v8_lowrank_vv — Recurrent low-rank V x V linear layer in vocab space

## Mechanism

Vocab-dim state (V = 1024). Input is one-hot. No output projection.

Per recurrent step:
- Cross-position: decay-weighted linear attention with per-dimension
  diagonal Q/K scaling (no full Q/K projections). Content similarity is
  computed directly between register states.
  ```
  scores[t, s] = (x_t * q_scale) . (x_s * k_scale)   causal, decay-masked
  out = scores @ x
  ```
- Within-position: low-rank V x V linear layer
  ```
  W_effective = U @ V^T + diag(d)      # U, V are (V, r)
  y = gelu(x @ U @ V^T + x * diag + bias) * out_scale
  ```

This is the best-performing variant in the benchmark at rank = 8.

## Rank behavior

| Rank | Params | Behavior |
|---|---|---|
| 8 | 164K | val_loss 5.24 at step 100, train ~ val, generalizing |
| 64 | 1.1M | train loss 0.04, val stays high, memorizing |

The rank-64 low-rank factor has enough capacity to store a bigram
lookup table. Rank 8 does not, which forces it to learn a compressed,
generalizable word-to-word interaction instead. The constraint is doing
the work.

## Honest framing

`x @ U @ V^T + x * diag` is mathematically a rank-r + diagonal linear
layer. The "word graph" framing that was originally attached to this
variant is decorative — the computation is a standard low-rank linear
map. What's actually useful here is that **direct bilinear
dimension-to-dimension interaction** is a good inductive bias when the
dimensions are vocabulary entries.

Cross-position cost is O(T^2 * V) — full attention cost in vocab space.
Does not scale to long sequences.

## Run

```bash
INTERACTION_RANK=8 MODEL_VERSION=v8_lowrank_vv \
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m glassbox_lm.training
```
