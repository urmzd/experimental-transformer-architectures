#!/usr/bin/env bash
# Setup glassbox-lm on a fresh machine (e.g. RunPod)
# Usage: bash setup.sh
set -euo pipefail

cd /workspace

# Clone if needed
[ -d glassbox-lm ] || git clone https://github.com/urmzd/glassbox-lm.git
cd glassbox-lm

# Install deps into system Python (torchrun uses system Python, not venv)
uv pip install --system -e .

# Download data
python data/download_data.py --variant sp1024

echo "Setup complete. Run training with:"
echo "  cd /workspace/glassbox-lm"
echo "  MODEL_VERSION=v8_lowrank_vv torchrun --standalone --nproc_per_node=\$(nvidia-smi -L | wc -l) train.py"
