#!/usr/bin/env bash
# Minimal CPU physics environment; GPU synthesis stays on pinned upstream deps.
set -euo pipefail
cd "$(dirname "$0")/.."
test -d third_party/Dexonomy/.git || bash scripts/setup_dexonomy.sh
actual_revision=$(git -C third_party/Dexonomy rev-parse HEAD)
test "$actual_revision" = d561ff9df8545deef7a3f9e7cfb4045ef66f6913
uv venv --allow-existing .venv-dexonomy-eval --python 3.11
uv pip install --python .venv-dexonomy-eval/bin/python -e '.[dev]' \
  mujoco==3.16.0 numpy==2.2.6 trimesh==4.5.3 transforms3d==0.4.2 \
  ruamel.yaml==0.18.10 omegaconf==2.3.0
# Only upstream's sim and scene-reading modules are used; no GPU/torch needed.
uv pip install --python .venv-dexonomy-eval/bin/python --no-deps -e third_party/Dexonomy
.venv-dexonomy-eval/bin/python -c 'import mujoco; from dexonomy.sim import MuJoCo_EvalEnv; assert hasattr(mujoco.mjtIntegrator,"mjINT_DISCRETE"); print(mujoco.__version__)'
