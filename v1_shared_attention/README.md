# v1: Shared Attention + Fourier Register Ops

**Status**: Best so far — loss dropped 7.42 → 4.78, val_bpb 2.83 in 250 steps on 3xA40.

## Architecture
- Shared causal self-attention (GQA, RoPE) reused across N recurrent steps
- Per-step Fourier register ops for within-position transforms
- dim = vocab_size = 1024, no embedding, no output projection

## Params
- ~3.2M total (99% in shared attention Q/K/V/O matrices)
- 8 recurrent steps, 32 channels, 16 Fourier basis

## Results (3xA40, 10min)
```
step:0   val_bpb:4.3945
step:250 val_bpb:2.8318  (stopped at 10min wallclock)
```

## Run
```bash
TRAIN_BATCH_TOKENS=491520 GRAD_ACCUM_STEPS=16 NUM_RECURRENT_STEPS=8 \
torchrun --standalone --nproc_per_node=3 train.py
```
