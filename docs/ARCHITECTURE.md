# Architecture — how this system is organized

This is the map of the system: what the project is trying to do, how the code
is laid out, what the shared infrastructure provides, and the contract every
model variant follows. For *why* the project exists (the observability
problem), read [OBSERVABILITY.md](OBSERVABILITY.md) first. For how the pieces
are verified, see [TESTING.md](TESTING.md).

## What we're trying to do

Build language models whose computation is **readable by construction and as
good as an opaque model**. Every architecture here shares one mechanism:
`hidden_dim = vocab_size`, no learned embedding, no output projection — so the
register state at every step *is* a distribution over words. The bar is
**observability at no performance cost**: a readable model must match the
opaque-embedding baseline (`v13_with_embedding`), measured in bits-per-byte
against the [Parameter Golf](https://github.com/openai/parameter-golf) yardstick
(16 MB artifact, <10 min on 8× H100, FineWeb bpb).

Two research fronts, pursued together (details and current standing in
[`TODO.md`](../TODO.md) and the root [`README.md`](../README.md)):

- **Observability** — is the readable state real? Measured with the `observe`
  toolkit (causality, coverage, induction).
- **Performance** — close the readable-vs-opaque bpb gap without losing
  readability.

## Repository layout

| Path | Role |
|---|---|
| `core/` | Shared infrastructure: config, data loading, eval, quantization, model registry, base class |
| `vN_mechanism/` | One model variant per directory (`__init__.py` + `model.py`), auto-discovered |
| `train.py` | The single training entry point (torchrun/DDP) |
| `apps/cli/` | Installed console scripts: `benchmark` (GPU comparison) and `observe` (interpretability toolkit) |
| `microbench.py` | Synthetic-data CPU microbench (speed / gradient health only — never trained quality) |
| `results.py` | Aggregates `logs/*_manifest.json` into a results table (mean±std across seeds) |
| `run_all.py` | Trains every registered variant sequentially (registry-driven) |
| `scripts/` | Reproduction scripts (e.g. `rank_sweep.sh` regenerates the width-sweep table) |
| `data/` | Dataset/tokenizer download and preparation; shards land in `data/datasets/` (gitignored) |
| `tests/` | CPU test suite; parametrized over the registry so new variants are auto-covered |
| `artifacts/` | Committed run manifests backing published numbers; checkpoints are gitignored |
| `logs/` | Per-run outputs: manifests, checkpoints, quantized artifacts (gitignored) |
| `docs/` | This documentation set — see [docs/README.md](README.md) |
| `skills/` | Portable agent skill mirroring the repo conventions |

## The shared computation skeleton

Every variant implements the same outer loop; only the two mixing mechanisms
differ:

```
Input:  one-hot(token) → register state R ∈ ℝ^V   (V = vocab_size = 1024)
Repeat NUM_STEPS times:
  1. Cross-position mixing    (how positions interact: conv, linear attention, decay memory, …)
  2. Within-position transform (how vocab dims combine: MLP, low-rank V×V, Fourier mix, …)
Output: R → logit softcap → cross-entropy loss
```

No embedding, no output projection. The sole exception is
`v13_with_embedding`, the opaque baseline kept only to measure what
readability costs — never use it as a template.

## Core modules (`core/`)

| Module | Responsibility |
|---|---|
| `base.py` | `AgiModel` abstract base: version metadata, `Settings` inner class, `build_kwargs()` (maps config fields to constructor args), abstract `forward(input_ids, target_ids) → loss` |
| `registry.py` | Auto-discovery: imports every `v*/model.py` at the repo root and registers any `AgiModel` subclass with a `version` set; `build_model(version, args)` instantiates by name |
| `config.py` | `Hyperparameters`: grouped `pydantic-settings` classes, every field an env var with a default. Attribute lookup falls through the groups, so `args.lr` works regardless of which group owns `lr` |
| `data.py` | Token shard loading (`.bin`: 256-int32 header, magic `20240520`, uint16 tokens), `TokenStream` (sequential, crosses shard boundaries), `DistributedTokenLoader` (per-rank slicing) |
| `eval.py` | Validation loss and tokenizer-agnostic bits-per-byte via sentencepiece byte-length lookup tables; distributed reduction |
| `quantize.py` | int8 quantization for the 16 MB artifact, plus `CONTROL_TENSOR_NAME_PATTERNS` / `is_control_tensor` — the fp32-under-bf16 classification (see precision regime below) |

## The model-variant pattern

A variant is a directory `vN_mechanism_description/` (name the computation,
not a metaphor) containing `model.py` with one class:

```python
class MyModel(AgiModel):
    version = "vN_mechanism_description"   # matches the directory suffix

    class Settings(CommonSettings):        # optional; fields become env vars
        my_knob: int = 8

    def forward(self, input_ids, target_ids) -> Tensor:  # returns scalar loss
        ...
```

What the shared infrastructure gives you for free:

- **Discovery** — `core/registry.py` finds the class; `train.py`, `benchmark`,
  `run_all.py`, `observe`, and the whole test suite pick it up with no list to
  edit anywhere.
- **Configuration** — add new hyperparameters to `core/config.py` as
  env-var-backed fields with defaults. Never change defaults to run an
  experiment; override at runtime (`MY_KNOB=16 torchrun … train.py`).
- **Constructor wiring** — `build_kwargs()` filters config fields to what your
  `__init__` accepts; override it only to rename fields.

What you must get right yourself:

- **Precision classification** — see the next section.
- **Causality** — no information may flow from future positions;
  `tests/test_models.py` gates this.
- **State readability** — the register state stays in vocab space so
  `observe` can capture it (`tests/test_observe.py` gates capture inertness).

## Precision regime (bf16 + fp32 control tensors)

Models are initialized in fp32, cast to bf16, then **control tensors**
(scales, biases, decay logits, gates — anything whose name suffix-matches
`CONTROL_TENSOR_NAME_PATTERNS`, plus every param with `ndim < 2`) are cast
back to fp32. Forward runs under bf16 autocast; the `.float()` calls inside
model forwards are intentional upcasts for numerical stability.

The classification is **suffix match only**. Never add a pattern that is a
suffix of ordinary projection weight names (e.g. `weight`) — a historical
substring rule once matched every `nn.Linear` weight and silently trained
whole models in fp32. `tests/test_precision.py` pins this.

## Entry points and data flow

```
data/download_data.py ──► data/datasets/*.bin ─┐
                                               ▼
run_all.py / benchmark ──(env vars + torchrun)──► train.py ──► logs/<RUN_ID>_manifest.json
                                                    │              │        + checkpoint (.pt)
                                                    ▼              ▼        + int8+zlib artifact
                                              core/eval.py     results.py (table, mean±std)
                                                               observe    (interpretability probes
                                                                           on the checkpoint, CPU fp32)
```

- `train.py` — torchrun/DDP, wallclock- or iteration-budgeted, bf16 regime,
  checkpoint save/resume, int8+zlib artifact emission. Warmup steps run
  forward+backward only (no optimizer updates) so nothing trains outside the
  timed budget.
- `benchmark` (`apps/cli/benchmark.py`) — runs `train.py` per version under
  identical wallclock/batch conditions, per seed (`--seeds 1337,1338,1339`),
  and collects manifests. This is the harness that ranks trained quality.
- `microbench.py` — CPU, synthetic tokens, forward+backward only. Use it for
  speed, init loss, and gradient health; never to rank trained quality.
- `observe` (`apps/cli/observe.py`) — CPU/fp32 interpretability probes on
  trained checkpoints (`trace`, `wordmap`, `causality`, `demo`, `sweep`,
  `coverage`, `induction`); subcommand table in the root README.

## Provenance and results discipline

Every run writes `logs/<RUN_ID>_manifest.json` recording `seed`, `git_sha`,
`torch_version`, `cuda_version`, `gpu_name`, `protocol` (`wallclock-Ns` vs
`fixed-N-iter`), `tokens_seen`, and the full config dump. Published numbers
must trace to a **committed manifest** in `artifacts/` — claims that match no
committed manifest get marked "under revision" and re-run (see the 2026-06-09
methodology notes in the README). Multi-seed runs are aggregated by
`results.py` as mean±std per config.
