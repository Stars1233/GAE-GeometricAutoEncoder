#!/usr/bin/env bash
# Optional, isolated Hopper BF16 forward backend. Does not replace torch/FA2.
set -euo pipefail
GAE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FA3_ROOT="${1:-$GAE_ROOT/results/flash-attention-5231}"
FA3_PYTHON="${GAE_PYTHON:-${GAE_VENV:-$GAE_ROOT/.venv}/bin/python}"
FA3_REVISION=5231d95fe13733fb534c01895f7ea88c6a6c7793
if [[ ! -d "$FA3_ROOT/.git" ]]; then
  git clone --depth 1 --branch v2.7.4.post1 --recursive --shallow-submodules \
    https://github.com/Dao-AILab/flash-attention.git "$FA3_ROOT"
fi
[[ "$(git -C "$FA3_ROOT" rev-parse HEAD)" == "$FA3_REVISION" ]]
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$(dirname "$FA3_PYTHON"):$PATH"
export MAX_JOBS=2 NVCC_THREADS=2
for feature in BACKWARD SPLIT PAGEDKV APPENDKV LOCAL SOFTCAP PACKGQA FP16 FP8 VARLEN HDIM96 HDIM192 HDIM256 SM80; do
  export "FLASH_ATTENTION_DISABLE_${feature}=TRUE"
done
uv pip install --python "$FA3_PYTHON" ninja wheel
# This pinned version otherwise downloads NVCC 12.3, despite installed 12.8.
# Keep its CUDA >=12.3 validation; alter only compiler download/selection.
"$FA3_PYTHON" - "$FA3_ROOT" <<'PY'
from pathlib import Path
import subprocess,sys
root=Path(sys.argv[1])
s=subprocess.check_output(['git','-C',str(root),'show','HEAD:hopper/setup.py'],text=True)
start=s.index('    if bare_metal_version != Version("12.3"):')
end=s.index('    cc_flag = []',start)
s=s[:start]+'    # GAE: use the verified CUDA_HOME toolchain, without downloading another.\n\n'+s[end:]
(root/'hopper/setup.py').write_text(s)
PY
cd "$FA3_ROOT/hopper"
"$FA3_PYTHON" setup.py build_ext --inplace
LD_PRELOAD="$CUDA_HOME/lib64/libcudart.so.12" "$FA3_PYTHON" - <<'PY'
import torch,flash_attn_interface,flash_attn_3_cuda
print('FA3_IMPORT_OK',torch.__version__,torch.version.cuda)
PY
