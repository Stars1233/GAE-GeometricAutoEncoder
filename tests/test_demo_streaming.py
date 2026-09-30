"""Completed RGB is delivered before postprocessing, and survives preview failure."""

import ast
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import types
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_functions():
    # Exercise the actual UI functions without loading Gradio/ML dependencies.
    tree = ast.parse((ROOT / "camera_studio.py").read_text())
    selected = {"_run_i2v_stream", "generate_i2v", "_latest", "_duration_i2v"}
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    for n in defs:
        n.decorator_list = []
    namespace = {
        "Path": Path,
        "queue": queue,
        "signal": signal,
        "subprocess": subprocess,
        "sys": sys,
        "threading": threading,
        "time": time,
        "os": os,
        "json": json,
        "uuid": uuid,
        "ROOT": ROOT,
        "RESIDENT_ENGINE": None,
        "gr": types.SimpleNamespace(Error=RuntimeError),
    }
    exec(compile(ast.Module(body=defs, type_ignores=[]), "camera_studio.py", "exec"), namespace)  # noqa: S102 - test trusted local functions
    return namespace


def test_completed_rgb_arrives_before_process_exit(tmp_path):
    ns = load_functions()
    video = tmp_path / "000_pred.mp4"
    code = f"import json,time; print('GAE_ARTIFACT_READY '+json.dumps({{'path':{str(video)!r}}}),flush=True); time.sleep(1)"
    events = ns["_run_i2v_stream"]([sys.executable, "-c", code], tmp_path, 10)
    first = next(events)
    assert first["video"] == str(video)
    final = next(events)
    assert final["elapsed"] - first["elapsed"] > 0.7
    assert final["error"] is None
    with pytest.raises(StopIteration):
        next(events)


@pytest.mark.parametrize("fail", [False, True])
def test_only_paired_result_is_published(tmp_path, monkeypatch, fail):
    ns = load_functions()
    ns.update(
        OUTPUT_ROOT=tmp_path,
        HF_REPO="test",
        CKPT_DIR=tmp_path / "ckpt",
        DEFAULT_EXAMPLE_POSES=tmp_path / "absent.npz",
        _pose_reference_for_image=lambda _: None,
    )
    calls = []

    def run(command, directory, timeout):
        directory.mkdir()
        (directory / "000_pred.mp4").write_bytes(b"video")
        (directory / "000_preview.npz").write_bytes(b"geometry")
        yield {"video": str(directory / "000_pred.mp4"), "elapsed": 1, "error": None}
        yield {"video": None, "elapsed": 2, "error": None}

    def render(path, output=None, **kwargs):
        calls.append(path)
        if fail:
            raise ValueError("bad geometry")
        return path.with_suffix(".mp4"), {"total_seconds": 0.5}

    module = types.ModuleType("scripts.demo.progressive_preview")
    module.render_preview = render
    monkeypatch.setitem(sys.modules, "scripts.demo.progressive_preview", module)
    ns.update(_run_i2v_stream=run)
    gen = ns["generate_i2v"]("image.jpg", "scene", "forward", 17, 25, 2, 42, 2)
    assert len(next(gen)) == 8
    first = next(gen)
    assert first[0] is None and first[6] is None and not calls
    before_render = next(gen)
    assert before_render[0] is None and before_render[6] is None and not calls
    final = next(gen)
    assert len(final) == 8 and final[0].endswith("_pred.mp4") and len(calls) == 1
    assert (final[6] is None) == fail
    assert ("preview failed" in final[5]) == fail


def test_resident_engine_keeps_generation_settings(tmp_path):
    ns = load_functions()
    calls = []

    class FakeEngine:
        def generate(self, image, prompt, **kwargs):
            calls.append((image, prompt, kwargs))
            return dict(
                rgb_video="rgb.mp4",
                synchronized_video="paired.mp4",
                geometry="geometry.npz",
                timings={"request_seconds": 44.2},
            )

    ns.update(
        RESIDENT_ENGINE=FakeEngine(),
        OUTPUT_ROOT=tmp_path,
        HF_REPO="test",
        CKPT_DIR=tmp_path / "ckpt",
        DEFAULT_EXAMPLE_POSES=tmp_path / "none.npz",
        _pose_reference_for_image=lambda _: tmp_path / "poses.npz",
    )
    gen = ns["generate_i2v"]("image.jpg", "prompt", "example", 81, 25, 2, 123, 2)
    assert next(gen)[6] is None
    result = next(gen)
    assert result[6] == "paired.mp4"
    assert len(calls) == 1
    options = calls[0][2]
    assert (options["views"], options["steps"], options["cfg_scale"], options["seed"]) == (
        81,
        25,
        2,
        123,
    )
    assert options["poses"] == tmp_path / "poses.npz"
    with pytest.raises(StopIteration):
        next(gen)
