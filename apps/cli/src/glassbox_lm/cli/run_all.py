"""Run all model versions sequentially and collect results.

Versions come from registry auto-discovery, so new models are picked up
without editing this file. Per-version env overrides and exclusions below.
"""
import os
import subprocess

from glassbox_lm.core.registry import get_registry

# The opaque-embedding baseline is a control, not a contestant — run it explicitly.
EXCLUDE = {"v13_with_embedding"}

# Per-version hyperparameter overrides (env vars, applied on top of the shared config).
ENV_OVERRIDES = {
    "v1_shared_attn":  {"NUM_STEPS": "8"},
    "v2_conv":         {"NUM_STEPS": "16"},
    "v5_fft_linattn":  {"N_FOURIER_BASIS": "64"},
    "v7_soft_ops":     {"NUM_STEPS": "16", "N_CHANNELS": "64"},
}

# Detect GPU count
try:
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    n_gpus = len(result.stdout.strip().splitlines()) if result.returncode == 0 else 1
except FileNotFoundError:
    n_gpus = 1


def main(argv: list[str] | None = None):
    batch = os.environ.get("TRAIN_BATCH_TOKENS", "491520")
    grad_accum = os.environ.get("GRAD_ACCUM_STEPS", "16")
    log_every = os.environ.get("TRAIN_LOG_EVERY", "50")
    iterations = os.environ.get("ITERATIONS", "500")

    versions = [v for v in sorted(get_registry()) if v not in EXCLUDE]

    results = []
    for version in versions:
        run_id = f"{version}_eval"
        print(f"\n{'='*60}")
        print(f"  Running {version} (run_id={run_id})")
        print(f"{'='*60}\n")

        env = {
            **os.environ,
            "MODEL_VERSION": version,
            "TRAIN_BATCH_TOKENS": batch,
            "GRAD_ACCUM_STEPS": grad_accum,
            "TRAIN_LOG_EVERY": log_every,
            "ITERATIONS": iterations,
            "RUN_ID": run_id,
            **ENV_OVERRIDES.get(version, {}),
        }

        cmd = [
            "torchrun", "--standalone",
            f"--nproc_per_node={n_gpus}",
            "-m", "glassbox_lm.training",
        ]

        ret = subprocess.run(cmd, env=env)
        status = "OK" if ret.returncode == 0 else f"FAIL({ret.returncode})"
        results.append((version, status))
        print(f"\n  {version}: {status}\n")

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for version, status in results:
        print(f"  {version:10s} {status}")

    # Print results table
    print(f"\n{'='*60}")
    print("  Results Table")
    print(f"{'='*60}\n")
    from glassbox_lm.cli.results import main as results_main
    results_main([])


if __name__ == "__main__":
    main()
