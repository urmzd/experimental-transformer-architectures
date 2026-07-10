# v7_soft_ops — Differentiable register machine with soft addressing + soft op selection

## Mechanism

Vocab-dim state (V = 1024). Input is one-hot. No output projection.

Per recurrent step:
1. **Read** — soft-select source registers via a learned softmax over V
   dimensions; output is a weighted sum of register values.
2. **Select** — pick one op from a shared bank of N_OPS primitives
   (identity, relu, gelu, square, negate, abs, tanh, sigmoid) via
   Gumbel-softmax; soft during training, anneals toward one-hot.
3. **Apply** — run the (softly) selected op on the read values.
4. **Write** — soft-select destination registers via a second softmax
   over V; scatter the op output.

Cross-position mixing: linear attention with causal decay (same core
as v3 / v9).

## Status

Unstable. Two major loss spikes reported during training (9.35 at step
161, 8.28 at step 181) before recovery. The soft op-selection path is
the likely cause; see PonderNet-style temperature annealing or hard
routing (v11b_hard_routing) for alternatives.

Related: Neural Programmer / Neural Turing Machine family of
differentiable register machines.

## Run

```bash
MODEL_VERSION=v7_soft_ops \
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m glassbox_lm.training
```

## Env vars

| Variable | Default | Notes |
|---|---|---|
| `NUM_STEPS` | 16 | Number of instructions in the program |
| `N_CHANNELS` | 64 | Channel dim for ops |
| `N_OPS` | 8 | Op bank size |
