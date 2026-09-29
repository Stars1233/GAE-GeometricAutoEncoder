#!/usr/bin/env bash
# Gradio page for the Ulysses 81-view I2V demo. One request uses NGPU GPUs.
#
#   bash scripts/demo/run_serve_ulysses_i2v.sh
#   NGPU=8 PORT=7860 GRADIO_AUTH=demo:secret bash scripts/demo/run_serve_ulysses_i2v.sh
#
# GRADIO_SHARE=1 (default) prints a temporary public https link.
# Set GRADIO_SHARE=0 to only listen on 0.0.0.0:$PORT.
#
# koala Demo (full setup + launch; --host/--port are forwarded to the server):
#   koala submit -m demo --port 7860 --s3-log <request 8 GPUs> \
#     -c 'GRADIO_SHARE=0 bash /abs/path/to/scripts/demo/run_serve_ulysses_i2v.sh --host 0.0.0.0 --port 7860'
#   koala access JOB --ensure   # get/reuse the Web entry once Running
#   koala access JOB            # print the URL to share
# Notes: use an absolute path; koala --port must match the command --port; the
# command stays foreground (no auto-restart); default lease 24h (--lease 5d max).
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="$(pwd)"

NGPU="${NGPU:-8}"
export NGPU
VENV="${GAE_VENV:-/local-ssd/gae-venv}"
[[ -d /local-ssd ]] || VENV="${GAE_VENV:-/tmp/gae-venv}"
export HF_HOME="${HF_HOME:-$(dirname "$VENV")/hf-home}"
export PORT="${PORT:-7860}"
export GRADIO_SHARE="${GRADIO_SHARE:-1}"
PY="$VENV/bin/python"
# Keep weights and outputs outside the synced checkout so restarts reuse them.
STATE="$(dirname "$VENV")"
export GAE_CKPT_DIR="${GAE_CKPT_DIR:-$STATE/gae-ckpts}"
export GAE_SERVE_OUTPUT="${GAE_SERVE_OUTPUT:-$STATE/gae-serve-results}"

log() { echo "[serve_ulysses] $*"; }

if [[ "$ROOT" != /local-ssd/* && "$ROOT" != /tmp/* ]]; then
  LOCAL="${GAE_LOCAL_ROOT:-/local-ssd/GAE-GeometricAutoEncoder}"
  [[ -d /local-ssd ]] || LOCAL="${GAE_LOCAL_ROOT:-/tmp/GAE-GeometricAutoEncoder}"
  log "syncing checkout to $LOCAL"
  mkdir -p "$LOCAL"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete --no-owner --no-group \
      --exclude .git --exclude __pycache__ --exclude .venv \
      --exclude results --exclude ckpts --exclude '*.egg-info' --exclude .gradio_previews \
      "$ROOT/" "$LOCAL/"
  else
    tar -C "$ROOT" \
      --exclude .git --exclude __pycache__ --exclude .venv \
      --exclude results --exclude ckpts --exclude '*.egg-info' --exclude .gradio_previews \
      -cf - . | tar -C "$LOCAL" --no-same-owner -xf -
  fi
  [[ -f "$LOCAL/scripts/demo/serve_ulysses_i2v.py" ]] || { log "error: copy failed"; exit 1; }
  exec bash "$LOCAL/scripts/demo/run_serve_ulysses_i2v.sh" "$@"
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

if ! "$PY" -c "import torch, gae, omegaconf, cv2, gradio" >/dev/null 2>&1; then
  log "creating env at $VENV"
  [[ -x "$PY" ]] || uv venv "$VENV" --python 3.12
  uv pip install --python "$PY" -e "$ROOT" "gradio>=4.44"
fi

if ! "$PY" -c "import torch; raise SystemExit(0 if torch.version.cuda else 1)"; then
  log "replacing CPU torch with the CUDA 12.4 wheel"
  uv pip install --python "$PY" --reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple \
    torch==2.5.1 torchvision==0.20.1
fi

GOT="$("$PY" -c 'import torch; print(torch.cuda.device_count())')"
(( GOT >= NGPU )) || { log "error: need $NGPU GPUs, found $GOT"; exit 1; }

CAPTION_SRC="${CAPTION_SRC:-/threed-code/public_models/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
CAPTION_LOCAL="${CAPTION_LOCAL:-/local-ssd/qwen25-vl-7b}"
if [[ -d /local-ssd && -f "${CAPTION_SRC}/config.json" ]]; then
  if [[ ! -f "${CAPTION_LOCAL}/.copy_done" ]]; then
    log "copying Qwen2.5-VL-7B onto local ssd"
    mkdir -p "${CAPTION_LOCAL}"
    cp -aL "${CAPTION_SRC}/." "${CAPTION_LOCAL}/"
    touch "${CAPTION_LOCAL}/.copy_done"
  fi
  export CAPTION_MODEL="${CAPTION_LOCAL}"
fi

log "gpus=$NGPU port=$PORT share=$GRADIO_SHARE ckpts=$GAE_CKPT_DIR outputs=$GAE_SERVE_OUTPUT"
log "a public link is printed below when share=1; leave this process running"
# Forward any CLI flags (e.g. --host/--port from a koala Demo front command) to
# the server; with no flags it falls back to the PORT/HOST/GRADIO_SHARE env vars.
exec "$PY" scripts/demo/serve_ulysses_i2v.py "$@"
