#!/usr/bin/env bash
# Regenerate the docs/OBSERVABILITY.md width-sweep table end to end:
# v8_lowrank_vv at rank 8/32/64/128, 3 seeds each, 600s wallclock per run
# (the 1x H100 protocol the published table used), then the coverage probe
# (the observability axis) on every checkpoint. Aggregate bpb across seeds
# with `glassbox results` (prints mean±std per config).
#
# Requires a CUDA box with data downloaded (glassbox data download --variant sp1024).
set -euo pipefail
cd "$(dirname "$0")/.."

RANKS=${RANKS:-"8 32 64 128"}
SEEDS=${SEEDS:-"1337,1338,1339"}
MINUTES=${MINUTES:-10}
TOKENIZER=${TOKENIZER:-./data/tokenizers/fineweb_1024_bpe.model}

for rank in $RANKS; do
  out="logs/rank${rank}_sweep_results.json"
  INTERACTION_RANK=$rank glassbox benchmark \
    --versions v8_lowrank_vv --seeds "$SEEDS" --minutes "$MINUTES" \
    --output "$out"

  # INTERACTION_RANK must match the checkpoint or the U/V shapes fail to load.
  for ckpt in $(python -c "import json,sys; [print(r['model_path']) for r in json.load(open(sys.argv[1]))]" "$out"); do
    INTERACTION_RANK=$rank glassbox observe coverage \
      --checkpoint "$ckpt" --tokenizer "$TOKENIZER"
  done
done

glassbox results
