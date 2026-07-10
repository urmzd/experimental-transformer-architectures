"""Collect all manifest.json files from a logs directory and print a results table."""
import argparse
import json
import statistics
from pathlib import Path


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="glassbox results", description=__doc__)
    parser.add_argument("--logs", default="logs", help="directory holding *_manifest.json (default: logs)")
    args = parser.parse_args(argv)

    manifests = sorted(Path(args.logs).glob("*_manifest.json"))
    if not manifests:
        print(f"No manifests found in {args.logs}/")
        return

    runs = []
    for p in manifests:
        with open(p) as f:
            runs.append(json.load(f))

    # Sort by val_bpb (best first); runs without one go last
    runs.sort(key=lambda r: 999 if r.get("val_bpb") is None else r["val_bpb"])

    # Print markdown table
    print("| Run ID | Model | Seed | Protocol | Params | Size (int8) | val_bpb | val_loss | Steps | Train Time |")
    print("|--------|-------|------|----------|--------|-------------|---------|----------|-------|------------|")
    for r in runs:
        rid = r.get("run_id", "?")[:20]
        model = r.get("model_version", "?")
        seed = r.get("seed")
        protocol = r.get("protocol")
        params = r.get("params", 0)
        size = r.get("quantized_bytes", 0)
        bpb = r.get("val_bpb")
        loss = r.get("val_loss")
        steps = r.get("steps_trained", 0)
        time_s = (r.get("train_time_ms") or 0) / 1000

        params_str = f"{params/1e6:.1f}M" if params >= 1e6 else f"{params/1e3:.0f}K"
        size_str = f"{size/1e6:.1f}MB" if size >= 1e6 else f"{size/1e3:.0f}KB"
        bpb_str = f"{bpb:.4f}" if bpb is not None else "—"
        loss_str = f"{loss:.4f}" if loss is not None else "—"
        seed_str = str(seed) if seed is not None else "—"
        protocol_str = protocol if protocol is not None else "—"
        time_str = f"{time_s:.0f}s"

        print(f"| {rid} | {model} | {seed_str} | {protocol_str} | {params_str} | {size_str} | {bpb_str} | {loss_str} | {steps} | {time_str} |")

    print_aggregate(runs)


def group_key(r: dict):
    """Runs are the same experiment if everything but the seed matches."""
    return (
        r.get("model_version"),
        r.get("protocol"),
        r.get("lr"),
        r.get("batch_tokens"),
        r.get("train_seq_len"),
        r.get("warmup_steps"),
        r.get("world_size"),
        r.get("params"),
    )


def print_aggregate(runs: list[dict]):
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        # Legacy manifests without seed/protocol provenance cannot be safely
        # grouped (aborted and full runs look identical), so skip them.
        if r.get("val_bpb") is None or r.get("seed") is None or r.get("protocol") is None:
            continue
        groups.setdefault(group_key(r), []).append(r)

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if not multi:
        return

    print("\nMulti-seed aggregates (same config, seed varies):\n")
    print("| Model | Protocol | n | val_bpb mean±std | Seeds |")
    print("|-------|----------|---|------------------|-------|")
    for key, rs in sorted(multi.items(), key=lambda kv: statistics.mean(r["val_bpb"] for r in kv[1])):
        model, protocol = key[0], key[1] if key[1] is not None else "—"
        bpbs = [r["val_bpb"] for r in rs]
        mean = statistics.mean(bpbs)
        std = statistics.stdev(bpbs)
        seeds = ", ".join(str(r.get("seed", "?")) for r in rs)
        print(f"| {model} | {protocol} | {len(rs)} | {mean:.4f}±{std:.4f} | {seeds} |")


if __name__ == "__main__":
    main()
