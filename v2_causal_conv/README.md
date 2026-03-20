# v2: Depthwise Causal Conv + Fourier Register Ops

**Status**: Slower and worse than v1. Abandoned.

## Architecture
- Per-step depthwise causal conv1d for cross-position mixing
- Per-step Fourier register ops for within-position transforms
- No attention, no embedding, no output projection

## Params
- ~1.3M total (48 steps × ~27K params each)

## Results (3xA40)
- 71K tok/s (vs 205K for v1) — 3x slower
- Loss 6.55 → 6.40 in 20 steps — barely learning
- Killed early

## Why it failed
- Depthwise conv has no cross-channel mixing — each word dim independent
- 48 sequential conv passes were slower than 8 shared attention passes
- LR=0.001 was too conservative (later bumped but not re-tested)
