# Testing — what is verified, and how

How correctness is enforced in this repo: the CPU test suite, what each gate
protects, and what still requires a GPU run to verify. System map in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Philosophy

- **Registry-parametrized.** Model-facing tests iterate over
  `core.registry.get_registry()`, so a new variant is covered by discovery,
  forward/backward, causality, precision, and capture-inertness tests the
  moment its class exists — no test edits required.
- **CPU-only and fast.** Tests run at tiny dimensions (`vocab_size=32`,
  `num_steps=2`) so the whole suite runs on a laptop and in CI. Trained
  *quality* is deliberately not a test concern — that's what the `benchmark`
  harness and committed manifests are for.
- **Gates encode past incidents.** Several tests exist because a specific bug
  shipped once (fp32-shadowing control-tensor patterns, capture hooks mutating
  the forward pass). Don't weaken them; they are the regression memory.

## Running

```bash
uv sync --group dev
uv run pytest -q          # full suite, CPU
uv run ruff check .       # lint (E701/E702/E741 are house style, ignored)
```

CI (`.github/workflows/ci.yml`) runs exactly these two on every PR to `main`.

## What each test file guards

| File | Guards |
|---|---|
| `test_registry.py` | Every `v*/model.py` imports and registers; every registered version instantiates; unknown versions raise with the available list |
| `test_models.py` | Forward returns a finite scalar loss; backward produces finite, non-dead gradients; **causal masking** — perturbing a future position must not change earlier positions' loss (parametrized over versions × positions); v15's straight-through top-k estimator |
| `test_precision.py` | The bf16/fp32 control-tensor regime: projection weights are *not* classified as control tensors (the historical `"weight"` substring bug), suffix-match semantics, and after the train-time cast no ≥2-dim non-control weight remains fp32 |
| `test_observe.py` | **Capture inertness**: recording register states for `observe` produces bit-identical states to an unhooked forward — instrumentation can never alter the computation it measures |
| `test_eval.py` | Bits-per-byte math against hand-computed golden values (sentencepiece byte-length LUTs, uniform-model bpb), `VAL_MAX_TOKENS` capping, undersized-batch rejection |
| `test_data.py` | Shard format (magic/version header) roundtrip and rejection, seq-len trimming, `TokenStream` crossing shard boundaries, distributed loader rank-disjointness and ordering |
| `test_quantize.py` | int8 quantize→dequantize accuracy and size accounting for the 16 MB artifact |
| `test_config.py` | Env-var override plumbing, defaults, JSON-serializable config dump (what the manifest records), unknown-field errors |

## What is NOT covered by tests (and how it's verified instead)

| Concern | Verified by |
|---|---|
| Trained quality (bpb) | `benchmark --seeds 1337,1338,1339` on GPU; manifests committed to `artifacts/`; aggregated with `results.py` (mean±std) |
| Training stability, throughput | Real training runs; `microbench.py` for init-time gradient health and speed only |
| Multi-GPU (DDP) behavior | RunPod runs (`torchrun --nproc_per_node=N`); loader rank-slicing logic is unit-tested, the collective path is not |
| Observability claims (coverage, faithfulness) | `observe demo` (planted-bigram control), `observe coverage` / `induction` on trained checkpoints; `scripts/rank_sweep.sh` for reproduction |

## Results discipline

A number is only citable if it traces to a committed manifest in `artifacts/`
(seed, git SHA, protocol, config dump). Anything published without one gets
flagged "under revision" and re-run — this happened at scale in the 2026-06-09
methodology correction; treat it as the standing rule, not a one-off.

## When adding a model

1. Run `uv run pytest -q` — discovery, forward/backward, causality, precision,
   and capture-inertness cover the new class automatically.
2. If the model adds scalar/gate/scale parameters, make sure their names end
   with a pattern in `CONTROL_TENSOR_NAME_PATTERNS` (`core/quantize.py`) or
   add a new suffix pattern — `test_precision.py` will fail if projection
   weights get caught, but it cannot know a *missing* pattern is missing;
   check `test_train_cast_leaves_no_fp32_weights` output for your params.
3. Model-specific mechanisms with subtle semantics (like v15's
   straight-through estimator) get their own targeted test in
   `tests/test_models.py`.
