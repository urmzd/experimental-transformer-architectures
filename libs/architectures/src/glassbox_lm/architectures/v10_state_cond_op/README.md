# v10_state_cond_op — State-conditioned op dispatch

## Mechanism

Vocab-dim state (V = 1024). Input is one-hot. No output projection.

Per recurrent step:

```
state_compressed = compress(registers)            # V -> d
read_w, op_w, write_w = policy_mlp(state_compressed)
selected = softmax(read_w) . registers            # soft READ from V
h        = op_bank(selected, softmax(op_w))       # soft OP from bank
delta    = softmax(write_w) * h                   # soft WRITE to V
registers = registers + delta
```

The policy MLP observes the current state and emits three distributions
that parameterize per-step read, op, and write routing. The op bank is
shared across steps. Cross-position context comes from decay-weighted
linear attention in the compressed state space (same family as v9).

This is a form of input-conditional mixture-of-experts operating in
vocab space.

## Contrast with nearby variants

| Variant | Op selection | Read/write addressing |
|---|---|---|
| v7_soft_ops | soft, Gumbel, per-step fixed | soft, input-independent |
| v10_state_cond_op | soft, emitted by policy MLP | soft, **state-dependent** |
| v11b_hard_routing | hard, Gumbel straight-through | hard per-token |

## Status

Untested on the benchmark.

## Run

```bash
MODEL_VERSION=v10_state_cond_op \
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m glassbox_lm.training
```

## Env vars

| Variable | Default | Notes |
|---|---|---|
| `STATE_DIM` | 64 | Compressed state dimension |
| `N_OPS` | 8 | Shared op bank size |
| `NUM_STEPS` | 8 | Recurrent steps |
