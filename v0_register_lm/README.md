# v0: RegisterLM (Learned Embeddings)

**Status**: Original prototype. Replaced by v1.

## Architecture
- Learned embedding (dim=256) + shared attention + Fourier register ops
- Separate output projection back to vocab
- NOT "registers are words" — uses opaque embedding space

## Params
- ~485K trainable, 708KB compressed

## Files
- `train.py` — PyTorch/CUDA training (originally `train_register_lm.py`)
- `train_mlx.py` — MLX version for Mac iteration
- `ablation.sh` — Comparison: RegisterLM vs SharedAttnOnly vs TinyGPT
