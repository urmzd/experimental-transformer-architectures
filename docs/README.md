# Documentation map

Start here. Each document owns one question; results and current standing live
in the root [README](../README.md).

## Orientation (read in this order)

| Document | Question it answers |
|---|---|
| [OBSERVABILITY.md](OBSERVABILITY.md) | **Why** — what the observability problem actually is (mechanism vs I/O boundary), and the honest catches |
| [ARCHITECTURE.md](ARCHITECTURE.md) | **How** — system layout, shared infrastructure (`glassbox_lm.core`), the model-variant contract, entry points, precision regime, provenance discipline |
| [TESTING.md](TESTING.md) | **How it's verified** — what each test gates, what needs a GPU run instead, results discipline |

## Working on the repo

| Document | Purpose |
|---|---|
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Setup, dev loop, add-a-model checklist, commit/PR conventions |
| [../AGENTS.md](../AGENTS.md) | Operational reference for agents and humans: env-var table, training commands, RunPod deployment, file conventions |
| [../TODO.md](../TODO.md) | The live research roadmap (both fronts: observability and performance) |

## Research notes and findings

| Document | Purpose |
|---|---|
| [INTERESTING_FINDINGS.md](INTERESTING_FINDINGS.md) | CPU microbench (`glassbox microbench`) results — speed and gradient health, **not** trained quality |
| [INTERESTING_RESEARCH.md](INTERESTING_RESEARCH.md) | Related work (LMGP, TPGs, Neural GPU, DEQ, Mamba/RWKV/Hyena) |
| [DESIGN_SHARED_MEMORY.md](DESIGN_SHARED_MEMORY.md) | Design notes for the v3/v4 shared-memory-bank family |

Trained results, benchmark tables, and the Parameter Golf standing are in the
root [README](../README.md); each variant directory may also carry its own
`README.md` with variant-specific notes.
