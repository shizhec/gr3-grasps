#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_root=${1:-outputs/dexonomy_fdh6_v2}
results_root=${2:-outputs/dexonomy_dynamics_audit}
mkdir -p "$results_root"
eval_python=.venv-dexonomy-eval/bin/python
export MUJOCO_GL=${MUJOCO_GL:-egl}
"$eval_python" scripts/validate_dexonomy_lift.py --output "$source_root" \
  --results-output "$results_root" --integrator implicitfast > "$results_root/implicitfast.log"
"$eval_python" scripts/validate_dexonomy_lift.py --output "$source_root" \
  --results-output "$results_root" --video > "$results_root/discrete.log"
"$eval_python" scripts/validate_dexonomy_lift.py --output "$source_root" \
  --results-output "$results_root" --timestep .00025 > "$results_root/half_step.log"
"$eval_python" scripts/validate_dexonomy_lift.py --output "$source_root" \
  --results-output "$results_root" --no-close > "$results_root/open_control.log"
"$eval_python" scripts/check_dexonomy_gr3_ik.py --source "$source_root" \
  --output "$results_root/gr3_ik" > "$results_root/gr3_ik.log"
uv pip freeze --python "$eval_python" > "$results_root/environment.txt"
"$eval_python" scripts/report_dexonomy_dynamics.py --output "$results_root"
