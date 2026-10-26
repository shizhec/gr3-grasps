#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=78fd485fa82d9b9a063fb4985e371814587e666a
if [ ! -d third_party/curobo/.git ]; then
  git clone https://github.com/NVlabs/curobo.git third_party/curobo
fi
actual_revision=$(git -C third_party/curobo rev-parse HEAD)
if [ "$actual_revision" != "$revision" ]; then
  if [ -n "$(git -C third_party/curobo status --porcelain)" ]; then
    echo 'The cuRobo checkout has changes; preserve them before selecting the pinned revision.' >&2
    exit 1
  fi
  git -C third_party/curobo fetch origin "$revision"
  git -C third_party/curobo checkout --detach "$revision"
fi
uv venv --allow-existing .venv-curobo --python 3.11
uv pip install --python .venv-curobo/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv-curobo/bin/python -e 'third_party/curobo[cu12]' -e '.[dev]' 'omegaconf>=2.3,<3' 'trimesh>=4,<5' 'huggingface-hub>=0.34,<1'
.venv-curobo/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
