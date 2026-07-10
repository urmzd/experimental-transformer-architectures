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

- **Observability** — is the readable state real? Measured with the
  `glassbox observe` toolkit (causality, coverage, induction).
- **Performance** — close the readable-vs-opaque bpb gap without losing
  readability.

## Repository layout

The repo is a uv workspace: the root `pyproject.toml` declares members `libs/*`
and `apps/*`, each with its own distribution.

| Path | Role |
|---|---|
| `libs/core/` | Shared infrastructure (`glassbox_lm.core`, dist `glassbox-lm-core`): config, data loading, eval, quantization, model registry, base class |
| `libs/architectures/` | The discoverable zoo of model variants (`glassbox_lm.architectures`, dist `glassbox-lm-architectures`): one variant per `vN_mechanism/` directory (`__init__.py` + `model.py`), auto-discovered |
| `libs/data/` | Dataset/tokenizer download and preparation (`glassbox_lm.data`, dist `glassbox-lm-data`) |
| `libs/training/` | The single training entry point (`glassbox_lm.training`, dist `glassbox-lm-training`; torchrun/DDP, run as `torchrun … -m glassbox_lm.training`) |
| `apps/cli/` | The `glassbox` console script (`glassbox_lm.cli`, dist `glassbox-lm-cli`): `list`, `train`, `data`, `benchmark` (GPU comparison), `observe` (interpretability toolkit), `microbench` (synthetic-data CPU microbench — speed / gradient health only, never trained quality), `results` (aggregates `logs/*_manifest.json` into a table, mean±std across seeds), `run-all` (trains every registered variant sequentially, registry-driven) |
| `scripts/` | Reproduction scripts (e.g. `rank_sweep.sh` regenerates the width-sweep table) |
| `data/` | Runtime data artifacts; shards land in `data/datasets/` (gitignored), `data/manifest.json` is tracked |
| `tests/` | CPU test suite (repo root); parametrized over the registry so new variants are auto-covered |
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

## Core modules (`glassbox_lm.core`)

| Module | Responsibility |
|---|---|
| `base.py` | `AgiModel` abstract base: version metadata, `Settings` inner class, `build_kwargs()` (maps config fields to constructor args), abstract `forward(input_ids, target_ids) → loss` |
| `registry.py` | Auto-discovery via the `glassbox_lm.architectures` entry-point group (the bundled zoo registers itself; third-party packages can add architectures the same way), with a direct-import fallback and a `glassbox_lm.core.register` decorator; `build_model(version, args)` instantiates by name |
| `config.py` | `Hyperparameters`: grouped `pydantic-settings` classes, every field an env var with a default. Attribute lookup falls through the groups, so `args.lr` works regardless of which group owns `lr` |
| `data.py` | Token shard loading (`.bin`: 256-int32 header, magic `20240520`, uint16 tokens), `TokenStream` (sequential, crosses shard boundaries), `DistributedTokenLoader` (per-rank slicing) |
| `eval.py` | Validation loss and tokenizer-agnostic bits-per-byte via sentencepiece byte-length lookup tables; distributed reduction |
| `quantize.py` | int8 quantization for the 16 MB artifact, plus `CONTROL_TENSOR_NAME_PATTERNS` / `is_control_tensor` — the fp32-under-bf16 classification (see precision regime below) |

## The model-variant pattern

A variant is a directory
`libs/architectures/src/glassbox_lm/architectures/vN_mechanism_description/`
(name the computation, not a metaphor) containing `model.py` with one class:

```python
class MyModel(AgiModel):
    version = "vN_mechanism_description"   # matches the directory suffix

    class Settings(CommonSettings):        # optional; fields become env vars
        my_knob: int = 8

    def forward(self, input_ids, target_ids) -> Tensor:  # returns scalar loss
        ...
```

What the shared infrastructure gives you for free:

- **Discovery** — `glassbox_lm.core.registry` finds the class; the training
  loop, `glassbox benchmark`, `glassbox run-all`, `glassbox observe`, and the
  whole test suite pick it up with no list to edit anywhere.
- **Configuration** — add new hyperparameters to `glassbox_lm.core.config` as
  env-var-backed fields with defaults. Never change defaults to run an
  experiment; override at runtime (`MY_KNOB=16 torchrun … -m glassbox_lm.training`).
- **Constructor wiring** — `build_kwargs()` filters config fields to what your
  `__init__` accepts; override it only to rename fields.

What you must get right yourself:

- **Precision classification** — see the next section.
- **Causality** — no information may flow from future positions;
  `tests/test_models.py` gates this.
- **State readability** — the register state stays in vocab space so
  `glassbox observe` can capture it (`tests/test_observe.py` gates capture inertness).

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
glassbox data download ──► data/datasets/*.bin ─┐
                                                ▼
glassbox run-all / glassbox benchmark ──(env vars + torchrun)──► glassbox_lm.training ──► logs/<RUN_ID>_manifest.json
                                                                       │                     │  + checkpoint (.pt)
                                                                       ▼                     ▼  + int8+zlib artifact
                                                            glassbox_lm.core.eval   glassbox results (table, mean±std)
                                                                                     glassbox observe (interpretability probes
                                                                                                       on the checkpoint, CPU fp32)
```

- `glassbox_lm.training` (`torchrun … -m glassbox_lm.training`) — torchrun/DDP,
  wallclock- or iteration-budgeted, bf16 regime,
  checkpoint save/resume, int8+zlib artifact emission. Warmup steps run
  forward+backward only (no optimizer updates) so nothing trains outside the
  timed budget.
- `glassbox benchmark` (`apps/cli/src/glassbox_lm/cli/benchmark.py`) — runs the
  training loop per version under
  identical wallclock/batch conditions, per seed (`--seeds 1337,1338,1339`),
  and collects manifests. This is the harness that ranks trained quality.
- `glassbox microbench` — CPU, synthetic tokens, forward+backward only. Use it for
  speed, init loss, and gradient health; never to rank trained quality.
- `glassbox observe` (`apps/cli/src/glassbox_lm/cli/observe.py`) — CPU/fp32
  interpretability probes on
  trained checkpoints (`trace`, `wordmap`, `causality`, `demo`, `sweep`,
  `coverage`, `induction`); subcommand table in the root README.

## Provenance and results discipline

Every run writes `logs/<RUN_ID>_manifest.json` recording `seed`, `git_sha`,
`torch_version`, `cuda_version`, `gpu_name`, `protocol` (`wallclock-Ns` vs
`fixed-N-iter`), `tokens_seen`, and the full config dump. Published numbers
must trace to a **committed manifest** in `artifacts/` — claims that match no
committed manifest get marked "under revision" and re-run (see the 2026-06-09
methodology notes in the README). Multi-seed runs are aggregated by
`glassbox results` as mean±std per config.
