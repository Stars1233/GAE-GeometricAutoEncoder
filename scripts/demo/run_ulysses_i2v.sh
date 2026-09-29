#!/usr/bin/env bash
# 81-view I2V on N GPUs with Ulysses sequence parallel, with timing.
#
#   bash scripts/demo/run_ulysses_i2v.sh
#   NGPU=4 SCENE=bedroom bash scripts/demo/run_ulysses_i2v.sh
#
# Env: NGPU (8), SCENE (forest_lake_trail), SCENES (comma list, overrides SCENE),
#      TOTAL_VIEWS (81), CFG_PARALLEL (0), SAMPLE_STEPS (50),
#      SAVE_VIDEOS_ONLY (0: also write point clouds; 1: keep only RGB and depth mp4),
#      TRAJECTORIES (empty: each scene's GT cameras; or forward,backward,turn_left,turn_right).
#      Several scenes or directions load Qwen3, the DiT and the VAE once.
#      OUT (results/<scene>_u<NGPU>[_cfgp][_sN]),
#      GAE_VENV (/local-ssd/gae-venv), HF_HOME (/local-ssd/hf-home), GAE_HF_REPO.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="$(pwd)"

NGPU="${NGPU:-8}"
SCENE="${SCENE:-forest_lake_trail}"
TOTAL_VIEWS="${TOTAL_VIEWS:-81}"
CFG_PARALLEL="${CFG_PARALLEL:-0}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SAVE_VIDEOS_ONLY="${SAVE_VIDEOS_ONLY:-0}"
TAG="u${NGPU}"
[[ "$CFG_PARALLEL" == "1" ]] && TAG="${TAG}_cfgp"
[[ "$SAMPLE_STEPS" != "50" ]] && TAG="${TAG}_s${SAMPLE_STEPS}"
HF_REPO="${GAE_HF_REPO:-TencentARC/GAE-D64-1B}"
VENV="${GAE_VENV:-/local-ssd/gae-venv}"
[[ -d /local-ssd ]] || VENV="${GAE_VENV:-/tmp/gae-venv}"
export HF_HOME="${HF_HOME:-$(dirname "$VENV")/hf-home}"
PY="$VENV/bin/python"

log() { echo "[ulysses_i2v] $*"; }

# /threed-code rejects the temp files setuptools writes for an editable install.
if [[ "$ROOT" != /local-ssd/* && "$ROOT" != /tmp/* ]]; then
  LOCAL="${GAE_LOCAL_ROOT:-/local-ssd/GAE-GeometricAutoEncoder}"
  [[ -d /local-ssd ]] || LOCAL="${GAE_LOCAL_ROOT:-/tmp/GAE-GeometricAutoEncoder}"
  log "copying checkout to $LOCAL"
  rm -rf "$LOCAL"
  mkdir -p "$LOCAL"
  tar -C "$ROOT" \
    --exclude .git --exclude __pycache__ --exclude .venv \
    --exclude results --exclude gae.egg-info --exclude '*.egg-info' \
    -cf - . | tar -C "$LOCAL" --no-same-owner -xf - || true
  [[ -f "$LOCAL/src/stage2/models/ulysses.py" ]] || { log "error: copy failed"; exit 1; }
  export GAE_SHARE_ROOT="$ROOT"
  exec bash "$LOCAL/scripts/demo/run_ulysses_i2v.sh"
fi

if [[ -n "${SCENES:-}" ]]; then
  IFS=',' read -r -a SCENE_LIST <<< "${SCENES}"
else
  SCENE_LIST=("${SCENE}")
fi
if [[ -n "${TRAJECTORIES:-}" ]]; then
  IFS=',' read -r -a TRAJ_LIST <<< "${TRAJECTORIES}"
else
  TRAJ_LIST=("")
fi

if ! command -v uv >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    log "installing uv"
    python3 -m pip install -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

if ! "$PY" -c "import torch, gae, omegaconf, cv2" >/dev/null 2>&1; then
  log "creating env at $VENV"
  [[ -x "$PY" ]] || uv venv "$VENV" --python 3.12
  uv pip install --python "$PY" -e "$ROOT"
fi

if ! "$PY" -c "import torch; raise SystemExit(0 if torch.version.cuda else 1)"; then
  log "replacing CPU torch with the CUDA 12.4 wheel"
  uv pip install --python "$PY" --reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple \
    torch==2.5.1 torchvision==0.20.1
fi

"$PY" - <<'PY'
import torch
n = torch.cuda.device_count()
print(f"[ulysses_i2v] torch {torch.__version__} cuda={torch.cuda.is_available()} gpus={n}", flush=True)
PY
GOT="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
(( GOT >= NGPU )) || { log "error: need $NGPU GPUs, found $GOT"; exit 1; }

BATCH_DIR="results/_batch_${TAG}_$$"
mkdir -p "$BATCH_DIR"
SCENES_CSV="$(IFS=,; echo "${SCENE_LIST[*]}")"
TRAJ_CSV="$(IFS=,; echo "${TRAJ_LIST[*]}")"
NJOBS="$(
  SCENES_CSV="$SCENES_CSV" TRAJ_CSV="$TRAJ_CSV" TAG="$TAG" OUT="${OUT:-}" \
  BATCH_JSON="$BATCH_DIR/jobs.json" "$PY" - <<'PY'
import json, os
scenes = [s.strip() for s in os.environ["SCENES_CSV"].split(",") if s.strip()]
trajs = [s.strip() for s in os.environ.get("TRAJ_CSV", "").split(",")]
if not trajs:
    trajs = [""]
tag = os.environ["TAG"]
out = os.environ.get("OUT", "")
single = len(scenes) == 1 and len(trajs) == 1 and trajs[0] == "" and bool(out)
jobs = []
for scene in scenes:
    for traj in trajs:
        suffix = f"{tag}_{traj}" if traj else tag
        jobs.append({
            "image": f"examples/scenes/{scene}.jpg",
            "prompt_file": f"examples/scenes/{scene}.txt",
            "trajectory": traj,
            "output": out if single else f"results/{scene}_{suffix}",
        })
if not jobs:
    raise SystemExit("no scenes to run")
with open(os.environ["BATCH_JSON"], "w") as handle:
    json.dump(jobs, handle, indent=2)
print(len(jobs))
PY
)"
log "jobs=$NJOBS views=$TOTAL_VIEWS gpus=$NGPU cfg_parallel=$CFG_PARALLEL steps=$SAMPLE_STEPS videos_only=$SAVE_VIDEOS_ONLY (weights load once)"
START=$(date +%s)
GEN_ARGS=(
  --batch-json "$BATCH_DIR/jobs.json"
  --hf-repo "$HF_REPO"
  --output "$BATCH_DIR"
  --total-views "$TOTAL_VIEWS"
  --ulysses-size "$NGPU"
  --sample-steps "$SAMPLE_STEPS"
)
[[ "$CFG_PARALLEL" == "1" ]] && GEN_ARGS+=(--cfg-parallel)
[[ "$SAVE_VIDEOS_ONLY" == "1" ]] && GEN_ARGS+=(--no-pointcloud)
"$PY" scripts/demo/generate.py "${GEN_ARGS[@]}" -- --timing-json "$BATCH_DIR/timing.json"
END=$(date +%s)
log "wall clock: $((END - START)) s (includes download, load and video write)"

"$PY" - "$BATCH_DIR" <<'PY'
import json, shutil, sys
from pathlib import Path
batch = Path(sys.argv[1])
jobs = json.loads((batch / "job_index.json").read_text())
src = batch / "scannetpp"
if not src.is_dir():
    raise SystemExit(f"missing generated videos in {src}")
for i, job in enumerate(jobs):
    dest = Path(job["output"])
    dest.mkdir(parents=True, exist_ok=True)
    prefix = f"{i:03d}_"
    found = False
    for path in src.glob(prefix + "*"):
        found = True
        rest = path.name[len(prefix):]
        target = dest / ("timing.json" if rest == "timing.json" else f"000_{rest}")
        shutil.copyfile(path, target)
    if not found:
        raise SystemExit(f"no outputs for job {i} ({job['output']}) under {src}")
    print(f"[ulysses_i2v] collected {job['output']}", flush=True)
PY

"$PY" - "$BATCH_DIR/jobs.json" <<'PY'
import json, sys
from pathlib import Path
for job in json.loads(Path(sys.argv[1]).read_text()):
    timing = Path(job["output"]) / "timing.json"
    if timing.is_file():
        print(f"[ulysses_i2v] compute timing {job['output']}: {timing.read_text().strip()}", flush=True)
PY

if [[ "$SAVE_VIDEOS_ONLY" == "1" ]]; then
  "$PY" - "$BATCH_DIR/jobs.json" <<'PY'
import json, sys
from pathlib import Path
for job in json.loads(Path(sys.argv[1]).read_text()):
    dest = Path(job["output"])
    if not dest.is_dir():
        continue
    for path in dest.rglob("*"):
        if path.is_file() and path.name not in ("timing.json",) and not path.name.endswith(("_pred.mp4", "_depth.mp4")):
            path.unlink()
PY
fi

if [[ -n "${GAE_SHARE_ROOT:-}" ]]; then
  "$PY" - "$BATCH_DIR/jobs.json" "$GAE_SHARE_ROOT" "$SAVE_VIDEOS_ONLY" <<'PY'
import json, shutil, sys
from pathlib import Path
jobs = json.loads(Path(sys.argv[1]).read_text())
share = Path(sys.argv[2])
videos_only = sys.argv[3] == "1"
keep = ("timing.json",)
for job in jobs:
    src = Path(job["output"])
    dest = share / src
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if not path.is_file():
            continue
        if videos_only and path.name not in keep and not path.name.endswith(("_pred.mp4", "_depth.mp4")):
            continue
        shutil.copyfile(path, dest / path.name)
    print(f"[ulysses_i2v] results copied to {dest}", flush=True)
PY
fi
rm -rf "$BATCH_DIR"
