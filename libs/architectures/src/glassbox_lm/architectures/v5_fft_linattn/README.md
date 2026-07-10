# v5_fft_linattn — Linear attention with causal decay, projections via rFFT

## Mechanism

Same linear-attention-with-decay core as v3_fourier_linattn, but V -> C
projections are computed directly via rFFT rather than through a stored
sin/cos basis matrix:

```
X = rfft(x, dim=-1)                 # O(V log V)
X = X[..., 1:n_freq+1]              # drop DC, keep n_freq harmonics
X_ri = concat(X.real, X.imag)       # 2*n_freq real features
channels = X_ri @ W.T               # learned linear map
```

Compared to v3:
- No (V, 2K) basis buffer.
- O(V log V) projection cost instead of O(V * K).
- Access to all harmonics up to `n_freq` (default 64) instead of the
  first 16.

## Status

Shape bug previously fixed. Mechanism has the same conceptual issue as
v3: the frequency decomposition is over arbitrary BPE vocab-id ordering,
which is a structural parameterization rather than a meaningful signal
basis. Not competitive on our benchmark.

## Run

```bash
MODEL_VERSION=v5_fft_linattn \
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m glassbox_lm.training
```
