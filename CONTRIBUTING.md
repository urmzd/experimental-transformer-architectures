# Contributing

Orientation: [docs/README.md](docs/README.md) maps all documentation.
The system design is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the
verification story is in [docs/TESTING.md](docs/TESTING.md); operational
commands and the env-var table are in [AGENTS.md](AGENTS.md).

## Setup

```bash
uv sync --group dev                          # deps + pytest/ruff (installs the `glassbox` CLI: `glassbox benchmark`/`glassbox observe`)
glassbox data download --variant sp1024      # only needed for real training/eval
```

## Dev loop

```bash
uv run pytest -q            # CPU test suite — must pass
uv run ruff check .         # lint — must pass (CI runs both on every PR)
glassbox microbench v8_lowrank_vv   # quick CPU sanity for a variant (speed/gradients, not quality)
```

Ground rules (the "why" for each is in [AGENTS.md](AGENTS.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

- **Never edit hyperparameter defaults** to run an experiment — every knob is
  an env var; override at runtime.
- **No embeddings, no output projections** — the register state stays in
  vocab space (only `v13_with_embedding` is exempt, as the opaque baseline).
- **Name mechanisms, not metaphors** — directories, versions, and classes
  describe the computation performed.
- **Numbers need manifests** — don't publish a result that doesn't trace to a
  committed manifest in `artifacts/`.

## Adding a model variant

1. Create `libs/architectures/src/glassbox_lm/architectures/vN_mechanism_description/`
   with `__init__.py` and `model.py`.
2. Subclass `AgiModel` (`glassbox_lm.core.base`): set `version` to match the directory
   suffix, implement `forward(input_ids, target_ids) → scalar loss`.
3. New hyperparameters go in `glassbox_lm.core.config` as env-var-backed fields with
   defaults.
4. Scalar/gate/scale parameters must stay fp32 under bf16: name them with a
   suffix in `CONTROL_TENSOR_NAME_PATTERNS` (`glassbox_lm.core.quantize`) or add a new
   pattern — never one that suffixes ordinary projection weight names.
5. Run `uv run pytest -q`. Discovery, forward/backward, causal masking,
   precision, and observe-capture tests pick the new version up automatically
   — `glassbox run-all`, `glassbox benchmark`, and `glassbox observe` need no edits either.
6. Document it: a row in the root README architecture table, a line in
   what-we've-learned once trained, and `TODO.md` if it opens work.

## Commits and PRs

- Conventional commits (`feat(observe): …`, `fix(v14): …`, `docs: …`,
  `chore: …`), imperative subject lines.
- PRs target `main`; CI (lint + CPU tests) must be green.
- Training results in a PR should include the manifest (commit it under
  `artifacts/` if the number is meant to be cited).
