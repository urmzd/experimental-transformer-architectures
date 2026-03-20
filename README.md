# RegisterGPT

A language model where **each register is a word**.

## The Idea

Standard transformers map tokens into opaque embedding spaces. RegisterGPT keeps computation in vocabulary space the entire time:

```
Input:  one-hot("cat") → R["cat"] = 1.0, all else 0.0
State:  always a distribution over words
Output: register state IS the prediction — R["dog"]=0.3, R["mat"]=0.25
```

No embedding matrix. No output projection. Every intermediate state is readable as "which words are active and how strongly."

## Architecture

```
token → one-hot(vocab_size) → [shared attention + Fourier register op] × N → logits
```

- **Shared self-attention** handles cross-position communication. Weights paid once, reused N times.
- **Fourier register ops** are tiny (~585 param) instructions that read/transform/write word activations using Fourier basis functions over vocabulary indices.
- **No embedding, no output projection.** The vocabulary IS the hidden dimension.

## Files

```
model.py                          # RegisterGPT architecture (clean, documented)
train.py                          # PyTorch/CUDA training (for GPU, torchrun compatible)
train_mlx.py                      # MLX training (for local Mac iteration)
experiments/
  train_register_lm.py            # Earlier version with learned embeddings (dim != vocab)
  ablation.sh                     # Comparison: RegisterLM vs SharedAttnOnly vs TinyGPT
```

## Usage

**GPU** (requires parameter-golf data in `./data/`):
```bash
torchrun --standalone --nproc_per_node=1 train.py
```

**Mac** (Apple Silicon, MLX):
```bash
ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 python3 train_mlx.py
```

## Context

Built for [OpenAI Parameter Golf](https://github.com/openai/parameter-golf). Inspired by [linear-gp](https://github.com/urmzd/linear-gp) — where complex behavior emerges from sequential execution of cheap operations on a narrow register bank.
