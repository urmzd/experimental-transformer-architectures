# v_gauss: FFT-Based Register Operations

**Status**: Design only, not yet trained.

## Architecture
- Replaces stored Fourier basis with `torch.fft.rfft` (computed on the fly)
- Uses all 512 frequencies instead of first 16
- O(V log V) projections instead of O(V × n_basis) matmuls
- Complex coefficients encode amplitude + phase

## Files
- `model.py` — GaussRegisterGPT model definition
- `GAUSS.md` — Design document: Gauss's math applied to RegisterGPT
