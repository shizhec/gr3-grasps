#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=d561ff9df8545deef7a3f9e7cfb4045ef66f6913
if [ ! -d third_party/Dexonomy/.git ]; then
  git clone https://github.com/JYChen18/Dexonomy.git third_party/Dexonomy
fi
actual_revision=$(git -C third_party/Dexonomy rev-parse HEAD)
if [ "$actual_revision" != "$revision" ]; then
  if [ -n "$(git -C third_party/Dexonomy status --porcelain)" ]; then
    echo 'Dexonomy checkout has local changes; preserve them before selecting the pinned revision.' >&2
    exit 1
  fi
  git -C third_party/Dexonomy fetch origin "$revision"
  git -C third_party/Dexonomy checkout --detach "$revision"
fi
uv venv --allow-existing .venv-dexonomy --python 3.11
uv pip install --python .venv-dexonomy/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv-dexonomy/bin/python -e third_party/Dexonomy -e '.[dev]'
.venv-dexonomy/bin/python -c 'import torch,mujoco; assert torch.cuda.is_available(); print(torch.__version__,mujoco.__version__,torch.cuda.get_device_name())'
