#!/usr/bin/env bash
# Resident 81-view service. Provision the environment/checkpoints before launch.
# Optional FA3: GAE_FA3_ROOT=/path/to/flash-attention-5231 bash scripts/demo/run_fast_demo.sh
set -euo pipefail
GAE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GAE_ROOT"
GAE_PYTHON="${GAE_VENV:-$GAE_ROOT/.venv}/bin/python"
[[ -x "$GAE_PYTHON" ]] || { echo 'Set GAE_VENV to the provisioned GAE environment.' >&2; exit 1; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export GAE_SPACE_OUTPUT_DIR="${GAE_SPACE_OUTPUT_DIR:-$GAE_ROOT/results/camera-studio}"
export GAE_CKPT_DIR="${GAE_CKPT_DIR:-$GAE_ROOT/ckpts}"
if [[ -n "${GAE_FA3_ROOT:-}" ]]; then
  [[ -f "$GAE_FA3_ROOT/hopper/flash_attn_interface.py" ]] || { echo 'Build the optional FA3 backend first.' >&2; exit 1; }
  export GAE_ATTENTION_BACKEND=fa3
  export PYTHONPATH="$GAE_FA3_ROOT/hopper${PYTHONPATH:+:$PYTHONPATH}"
  export LD_PRELOAD="${CUDA_HOME:-/usr/local/cuda-12.8}/lib64/libcudart.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
fi
GAE_STATE="${GAE_SERVER_STATE:-$GAE_ROOT/results/service-$(date +%Y%m%d-%H%M%S)-$$}"
exec "$GAE_PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node="${NGPU:-8}" \
  scripts/demo/distributed_engine.py --serve --cfg-parallel \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-7860}" \
  --checkpoint-dir "$GAE_CKPT_DIR" \
  --image examples/scenes/forest_lake_trail.jpg \
  --prompt-file examples/scenes/forest_lake_trail.txt \
  --poses examples/scenes/forest_lake_trail_poses.npz \
  --views 81 --steps 25 --warmup-steps 2 --output "$GAE_STATE" "$@"
