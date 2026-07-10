#!/usr/bin/env bash
# Bootstrap glassbox-lm on RunPod from scratch
# Usage: curl -sSL https://raw.githubusercontent.com/urmzd/glassbox-lm/main/bootstrap.sh | bash
set -euo pipefail

cd /workspace

# Clone repo
[ -d glassbox-lm ] || git clone https://github.com/urmzd/glassbox-lm.git
cd glassbox-lm

# Install uv
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Ensure uv is on PATH for this session and future shells
export PATH="$HOME/.local/bin:$PATH"
grep -q '.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Install deps into system Python (torchrun uses system Python, not venv)
uv pip install --system -e .

# Download data
python data/download_data.py --variant sp1024

echo ""
echo "=== Ready ==="
echo "cd /workspace/glassbox-lm"
echo ""
echo "# Run v8_lowrank_vv (best variant so far):"
echo "MODEL_VERSION=v8_lowrank_vv INTERACTION_RANK=8 \\"
echo "TRAIN_BATCH_TOKENS=491520 GRAD_ACCUM_STEPS=16 \\"
echo "TRAIN_LOG_EVERY=10 RUN_ID=v8_run \\"
echo "torchrun --standalone --nproc_per_node=\$(nvidia-smi -L | wc -l) train.py"
