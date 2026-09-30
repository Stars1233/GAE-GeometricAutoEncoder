"""Exercise editor interaction and every generated camera against Python."""
import shutil
import subprocess
from pathlib import Path
import pytest


def test_camera_editor_events_and_pose_parity():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is unavailable')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root/'tests/probes/check_camera_editor.cjs'), str(root)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
