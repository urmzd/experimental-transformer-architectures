#!/usr/bin/env python3
"""Collect all manifest.json files from logs/ and print a results table."""
import json
from pathlib import Path


def main():
    manifests = sorted(Path("logs").glob("*_manifest.json"))
    if not manifests:
        print("No manifests found in logs/")
        return

    runs = []
    for p in manifests:
        with open(p) as f:
            runs.append(json.load(f))

    # Sort by val_bpb (best first)
    runs.sort(key=lambda r: r.get("val_bpb") or 999)

    # Print markdown table
    print("| Run ID | Model | Params | Size (int8) | val_bpb | val_loss | Steps | Train Time |")
    print("|--------|-------|--------|-------------|---------|----------|-------|------------|")
    for r in runs:
        rid = r.get("run_id", "?")[:20]
        model = r.get("model_version", "?")
        params = r.get("params", 0)
        size = r.get("quantized_bytes", 0)
        bpb = r.get("val_bpb")
        loss = r.get("val_loss")
        steps = r.get("steps_trained", 0)
        time_s = (r.get("train_time_ms") or 0) / 1000

        params_str = f"{params/1e6:.1f}M" if params >= 1e6 else f"{params/1e3:.0f}K"
        size_str = f"{size/1e6:.1f}MB" if size >= 1e6 else f"{size/1e3:.0f}KB"
        bpb_str = f"{bpb:.4f}" if bpb else "—"
        loss_str = f"{loss:.4f}" if loss else "—"
        time_str = f"{time_s:.0f}s"

        print(f"| {rid} | {model} | {params_str} | {size_str} | {bpb_str} | {loss_str} | {steps} | {time_str} |")


if __name__ == "__main__":
    main()
