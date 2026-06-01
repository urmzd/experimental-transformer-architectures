# Experimental Transformer Architectures — Agent Guidelines

## Principles

- **Don't change defaults** — use environment variables to override hyperparameters at runtime, not by editing default values in code.
- **Self-contained repo** — `train.py` is the single entry point for training all models. Shared infrastructure lives in `core/` (config, data loading, eval, quantization, model registry). Model definitions live in their own directories (`v9_linattn/model.py`, etc.) and are auto-discovered by `core/registry.py` via the `version` class attribute.
- **No embedding, no output projection** — every model operates in vocabulary space. Input is one-hot, output is the register state. Keeping the whole computation in vocab space is what makes the state *observable* (every activation is a readable distribution over words) — that is the project's goal; do not add embedding layers or output projections. The sole exception is `v13_with_embedding`, an opaque-embedding baseline kept only for comparison — do not reuse it as a template.
- **Environment variables for everything** — all hyperparameters live in the `Hyperparameters` class in `core/config.py` and are read from env vars. When adding a new model, add its specific env vars there with sensible defaults.
- **Names describe mechanism, not metaphor** — variant directory names, version strings, and class names should describe the computation performed. Neuroscience / physics / information-theory names (brain waves, Gauss, Thousand Brains, Q-tables, TPGs) are decorative and should not be used as primary identifiers.

## Adding a new model version

1. Create a directory: `vN_mechanism_description/` where the suffix names the distinctive computation (e.g. `vN_linattn`, `vN_data_dependent`). Avoid metaphor.
2. Add `__init__.py` and `model.py` with a single model class.
3. The model class must implement `forward(input_ids: Tensor, target_ids: Tensor) -> Tensor` returning the loss.
4. Set `version = "vN_mechanism_description"` on the class (matches the directory suffix).
5. Add any new env vars to the appropriate config class in `core/config.py`.
6. Add any new control tensor name patterns to `CONTROL_TENSOR_NAME_PATTERNS` in `core/quantize.py` (these stay in fp32 during bfloat16 training).
7. Add the model to the `MODELS` list in `run_all.py`.
8. Update top-level `README.md` — add a row to the architecture table and a line to what-we've-learned.
9. Update `TODO.md` if relevant.

## Control tensor patterns

Parameters matching patterns in `CONTROL_TENSOR_NAME_PATTERNS` are kept in float32 even when the model is cast to bfloat16. This includes: scales, biases, decay logits, gating parameters, and small learned scalars. When adding a new model, ensure any scalar/gate/scale parameters have names matching existing patterns or add new patterns.

## Training conventions

- All models train via `torchrun --standalone --nproc_per_node=N train.py`.
- Multi-GPU via PyTorch DDP — batch size must be divisible by `num_gpus * GRAD_ACCUM_STEPS * TRAIN_SEQ_LEN(1024)`.
- Mixed precision: bfloat16 for weights, float32 for control tensors, autocast during forward.
- Models are initialized in float32, cast to bfloat16, then control tensors converted back to float32.
- The `.float()` calls inside model forward methods are intentional — they upcast for numerical stability before projections.

## RunPod deployment

SSH config alias: `runpod` (configured in `~/.ssh/config`, key: `~/.ssh/runpod`).

Setup on a fresh pod:
```bash
cd /workspace && \
git clone https://github.com/urmzd/experimental-transformer-architectures.git && \
cd experimental-transformer-architectures && bash setup.sh
```

Run training:
```bash
cd /workspace/experimental-transformer-architectures && \
TRAIN_BATCH_TOKENS=491520 \
GRAD_ACCUM_STEPS=16 \
TRAIN_LOG_EVERY=10 \
MODEL_VERSION=v9_linattn \
RUN_ID=<name> \
torchrun --standalone \
--nproc_per_node=$(nvidia-smi -L | wc -l) \
train.py
```

## Key env vars

| Variable | Default | Notes |
|---|---|---|
| `MODEL_VERSION` | `v8_lowrank_vv` | Which model to train (see table below) |
| `NUM_STEPS` | 8 | Recurrent steps / depth |
| `STATE_DIM` | 64 | State / linear-attention feature dim (v9+) |
| `INNER_DIM` | 128 | Inner MLP dimension (v9+) |
| `N_FOURIER_BASIS` | 16 | Fourier basis count (v1–v6) |
| `N_CHANNELS` | 128 | Channel dim (v1–v6) |
| `N_OPS` | 8 | Op bank size (v7, v10) |
| `K_ACTIVE` | 256 | Active register slice size (v12) |
| `GUMBEL_TAU` | 1.0 | Gumbel temperature (v11b_hard_routing) |
| `HALT_THRESHOLD` | 0.5 | Early-exit threshold (v11b_hard_routing) |
| `PONDER_LAMBDA` | 0.01 | Ponder regularization (v11b_hard_routing) |
| `LR` | 0.03 | Adam learning rate |
| `DECAY_INIT` | 3.0 | Memory decay logit |
| `GRAD_ACCUM_STEPS` | 16 | Gradient accumulation |
| `TRAIN_BATCH_TOKENS` | 524288 | Global batch size in tokens |
| `MAX_WALLCLOCK_SECONDS` | None | Wall-clock time limit (must be set manually, no default) |
| `VAL_MAX_TOKENS` | 0 | Cap validation to the first N tokens (0 = full set). Set >0 for fast eval during experimentation; leave 0 for a faithful bits-per-byte score |
| `ITERATIONS` | 500 | Max training iterations |
| `TORCH_COMPILE` | 0 | Enable torch.compile |
| `ROUNDTRIP_EVAL` | 0 | Run int8 quantization roundtrip eval after training |
| `NCCL_P2P_DISABLE` | 1 | Disable NCCL P2P; required on RunPod where GPUs span PCIe root complexes |

## File conventions

- `core/config.py` — all hyperparameter classes (`Hyperparameters`, `BaseSettings` subclasses)
- `core/data.py` — data loading (`TokenStream`, `DistributedTokenLoader`)
- `core/eval.py` — validation (`eval_val`, `build_sentencepiece_luts`)
- `core/quantize.py` — int8 quantization/dequantization, `CONTROL_TENSOR_NAME_PATTERNS`
- `core/registry.py` — model registry (auto-discovers `AgiModel` subclasses with a `version` set)
- `train.py` — training loop, DDP setup, checkpointing, serialization
- `apps/cli/benchmark.py` — synthetic data benchmarking (no GPU needed), tests all models
- `run_all.py` — sequential training of all models, results collection
- `results.py` — reads `logs/*_manifest.json` and prints a results table
- `tests/` — pytest tests for config, registry, quantization, and models
- Model directories are named `vN_mechanism_description/`
- Each model directory contains `__init__.py` and `model.py`
- Research notes go in `docs/`

## MODEL_VERSION values

| `MODEL_VERSION` | Directory | Mechanism |
|---|---|---|
| `v1_shared_attn` | `v1_shared_attention/` | Shared GQA attention + Fourier-parameterized channel mix |
| `v2_conv` | `v2_causal_conv/` | Depthwise causal conv + Fourier-parameterized channel mix |
| `v3_fourier_linattn` | `v3_fourier_linattn/` | Linear attn with causal decay; Q/K/V/O via Fourier basis |
| `v4_weight_shared` | `v4_weight_shared/` | Size-reduced v3 (shared Q/K, factored mix, step reuse) |
| `v5_fft_linattn` | `v5_fft_linattn/` | Linear attn with causal decay; Q/K/V/O via rFFT |
| `v6_banded_fourier` | `v6_banded_fourier/` | Band-partitioned Fourier linattn with gated coupling |
| `v7_soft_ops` | `v7_soft_ops/` | Soft op-bank + soft register addressing + linattn |
| `v8_lowrank_vv` | `v8_lowrank_vv/` | Recurrent rank-r V x V linear layer (best at rank 8) |
| `v9_linattn` | `v9_linattn/` | Linear attn with causal decay; dense projections |
| `v10_state_cond_op` | `v10_state_cond_op/` | State-conditioned soft read/op/write dispatch |
| `v11a_mixed_ops` | `v11a_mixed_ops/` | Five fixed primitive ops composed sequentially |
| `v11b_hard_routing` | `v11b_hard_routing/` | Hard Gumbel routing + multi-timescale linattn + halting |
| `v12_vocab_slice` | `v12_vocab_slice/` | Processing in fixed k-length vocab-id slices |
| `v13_with_embedding` | `v13_with_embedding/` | **Opaque-embedding baseline** — adds learned embedding (for comparison only) |
| `v14_data_dependent` | `v14_data_dependent/` | Mamba/RWKV/Hyena-style data-dependent dynamics |
| `v15_aux_loss` | `v15_aux_loss/` | v12 body + per-step CE + top-k + entropy-scaled writes |
| `v16_multi_branch` | `v16_multi_branch/` | Multi-column ensemble + branched gated MLP |
