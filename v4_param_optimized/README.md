# v4: Parameter-Golf-Optimized Register Machine

**Status**: Design only, not yet trained.

## Architecture
- Targets ~101K params (69% reduction from v3's 329K)
- Factored channel_mix: diagonal + low-rank
- Shared Q/K projections across all steps
- Multi-head decay (H=4) for multi-timescale retrieval
- Step reuse: 5 unique steps × 2 invocations = 10 depth

## Files
- `model.py` — Model definition only (no training script yet)
