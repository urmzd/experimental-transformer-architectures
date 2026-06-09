#!/usr/bin/env bash
# Setup experimental-transformer-architectures on a fresh machine (e.g. RunPod)
# Usage: bash setup.sh
set -euo pipefail

cd /workspace

# Clone if needed
[ -d experimental-transformer-architectures ] || git clone https://github.com/urmzd/experimental-transformer-architectures.git
cd experimental-transformer-architectures

# Install deps into system Python (torchrun uses system Python, not venv)
uv pip install --system -e .

# Download data
python data/download_data.py --variant sp1024

echo "Setup complete. Run training with:"
echo "  cd /workspace/experimental-transformer-architectures"
echo "  MODEL_VERSION=v8_lowrank_vv torchrun --standalone --nproc_per_node=\$(nvidia-smi -L | wc -l) train.py"
