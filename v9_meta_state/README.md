# v9: Meta-State — Q-table as Cross-Position Mechanism

**Status**: Best non-attention model — val_bpb **3.26** in 228 steps on 3×A40.

## Core Insight
The cross-position mechanism is a **Q-table that evolves during inference**.
Trained weights define the update rule. The table stores per-sequence knowledge.
Like RL's Q-table: starts empty, fills up through experience.

## Architecture
- Dense projections (nn.Linear, NOT Fourier) — full-rank, unconstrained
- Q-table accumulates key⊗value outer products causally
- Within-position: simple MLP bottleneck (down → gelu → up)
- No Fourier basis anywhere. Simple math only.

## What's different
| v3/v5/v6 | v9 (Meta-State) |
|---|---|
| Fourier projections (rank-32) | Dense projections (full-rank) |
| Static mechanism | Evolving Q-table |
| Weights = knowledge | Weights = update rule, table = knowledge |
| Complex parameterization | Simple: linear + gelu |

## Usage
```bash
MODEL_VERSION=meta TRAIN_BATCH_TOKENS=491520 GRAD_ACCUM_STEPS=16 \
TRAIN_LOG_EVERY=10 RUN_ID=meta_test \
torchrun --standalone --nproc_per_node=3 train.py
```

## Results (3×A40, 10min wallclock)
```
step:0   train_loss:~7.4
step:120 train_loss:5.74   (steady decline begins)
step:228 train_loss:5.50   val_bpb:3.2609  (stopped at wallclock limit)
```
- ~199K tok/s, ~2.64s/step
- 4.2M params, INT8 compressed to 8.9MB (1.66× ratio)

## Env Vars
| Variable | Default | Notes |
|----------|---------|-------|
| `STATE_DIM` | 64 | Q-table state dimension |
| `INNER_DIM` | 128 | MLP bottleneck width |
| `NUM_STEPS` | 8 | Number of query/update cycles |
