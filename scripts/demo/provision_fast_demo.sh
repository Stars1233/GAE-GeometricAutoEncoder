#!/usr/bin/env bash
# Provision the resident Camera Studio environment and public model caches.
set -euo pipefail
GAE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GAE_ROOT"
GAE_VENV="${GAE_VENV:-$GAE_ROOT/.venv}"
GAE_CKPT_DIR="${GAE_CKPT_DIR:-$GAE_ROOT/ckpts}"
if [[ ! -x "$GAE_VENV/bin/python" ]]; then
  uv venv --python 3.12 "$GAE_VENV"
fi
GAE_CONSTRAINTS="$(mktemp)"
trap 'rm -f "$GAE_CONSTRAINTS"' EXIT
cat > "$GAE_CONSTRAINTS" <<'CONSTRAINTS'
torch==2.5.1
torchvision==0.20.1
xformers==0.0.28.post3
numpy<2
transformers>=5,<6
gradio==6.28.0
CONSTRAINTS
uv pip install --python "$GAE_VENV/bin/python" -c "$GAE_CONSTRAINTS" -e '.[space]'
"$GAE_VENV/bin/python" scripts/demo/download_checkpoints.py --out-dir "$GAE_CKPT_DIR"
"$GAE_VENV/bin/python" - "$GAE_CKPT_DIR" <<'PY'
from pathlib import Path
import os
import sys
from gae.hub import extract_da3_stats
from huggingface_hub import snapshot_download
extract_da3_stats(Path(sys.argv[1]) / 'da3_stats_giant_5ds.tar', Path('model_stats/da3_giant_5ds'))
repos = ['depth-anything/DA3-GIANT-1.1', 'depth-anything/DA3METRIC-LARGE', 'Qwen/Qwen3-0.6B']
# Scene descriptions for uploaded images (~16 GB); GAE_CAPTION=0 skips it.
from scripts.demo.captioner import CAPTION_MODEL
if os.environ.get('GAE_CAPTION', '1') != '0' and not Path(CAPTION_MODEL).is_dir():
    repos.append(CAPTION_MODEL)
for repo in repos:
    snapshot_download(repo)
PY
