"""Browser demo for 81-view I2V on several GPUs.

Same sampler as ``scripts/demo/run_ulysses_i2v.sh``: one Gradio process queues
requests, and each request runs ``generate.py --ulysses-size N``, which
relaunches the sampler under torchrun. Weights load again per request.

    bash scripts/demo/run_serve_ulysses_i2v.sh

``GRADIO_SHARE=1`` (the default) prints a temporary public https link.
``GRADIO_AUTH=user:password`` asks for that password before anyone can generate.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.demo.trajectory_utils import (
    load_reference_poses, render_trajectory_preview, trajectory_for_preview,
)

HF_REPO = os.environ.get("GAE_HF_REPO", "TencentARC/GAE-D64-1B")
NGPU = max(1, int(os.environ.get("NGPU", "8")))
CKPT_DIR = Path(os.environ.get("GAE_CKPT_DIR", str(ROOT / "ckpts")))
OUTPUT_ROOT = Path(os.environ.get("GAE_SERVE_OUTPUT", str(ROOT / "results" / "ulysses_demo")))
TIMEOUT = int(os.environ.get("GEN_TIMEOUT", "3600"))
MIN_FREE_GIB = float(os.environ.get("MIN_FREE_GIB", "40"))
BOOT_TIMEOUT = int(os.environ.get("BOOT_TIMEOUT", "1800"))
WORKER_ROOT = Path(os.environ.get("GAE_WORKER_DIR", str(OUTPUT_ROOT.parent / "gae-serve-worker")))
CFG_PARALLEL = os.environ.get("CFG_PARALLEL", "1") == "1" and NGPU >= 2 and NGPU % 2 == 0
# Point-cloud subsample stride: larger => fewer points. Default 6 (was 4) to
# keep the previewed cloud lighter; override with PC_STRIDE=N.
PC_STRIDE = max(1, int(os.environ.get("PC_STRIDE", "6")))
RESOLUTION = (378, 672)
DEFAULT_FPS = 12
EXAMPLE_IMAGE = ROOT / "examples" / "scenes" / "forest_lake_trail.jpg"
WEB_FPS = int(os.environ.get("WEB_FPS", "0"))
DEFAULT_POSES = ROOT / "examples" / "scenes" / "forest_lake_trail_poses.npz"
CAPTION_MODEL = os.environ.get(
    "CAPTION_MODEL",
    "/threed-code/public_models/models--Qwen--Qwen2.5-VL-7B-Instruct"
    "/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
)
_LOCK = threading.Lock()
_CAPTION_LOCK = threading.Lock()
_captioner = None

_CAPTION_INSTRUCTION = (
    "Describe this photograph as one English paragraph for a video-generation prompt. "
    "Cover the place, objects, lighting, and atmosphere that are actually visible. "
    "Do not mention a camera, an image, or a prompt, and do not invent motion."
)

TRAJECTORIES = [
    ("Default camera poses", "example"),
    ("Forward — smooth forward move", "forward"),
    ("Backward — smooth backward move", "backward"),
    ("Turn left — gentle smooth rotation", "turn_left"),
    ("Turn right — gentle smooth rotation", "turn_right"),
]
VIEW_CHOICES = [17, 33, 81]


def _examples() -> list[list[str]]:
    rows = []
    for image in sorted((ROOT / "examples" / "scenes").glob("*.jpg")):
        prompt_file = image.with_suffix(".txt")
        if prompt_file.is_file():
            rows.append([str(image), prompt_file.read_text(encoding="utf-8").strip(), "example"])
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _example_index() -> dict[str, Path]:
    return {_sha256(image): image for image in (ROOT / "examples" / "scenes").glob("*.jpg")}


_EXAMPLES_BY_HASH = _example_index()


def _matching_example(image: str | None) -> Path | None:
    """Match uploads by content: Gradio keeps the client file name, so a user
    photo named ``bedroom.jpg`` must not pick up the bundled bedroom scene."""
    if not image:
        return None
    try:
        return _EXAMPLES_BY_HASH.get(_sha256(Path(image)))
    except OSError:
        return None


def _pose_file(image: str | None) -> Path | None:
    example = _matching_example(image)
    if example is not None:
        poses = example.with_name(f"{example.stem}_poses.npz")
        if poses.is_file():
            return poses
    return DEFAULT_POSES if DEFAULT_POSES.is_file() else None


def _first(root: Path, suffix: str) -> Path | None:
    files = sorted(p for p in root.rglob(f"*{suffix}") if p.is_file())
    return files[0] if files else None


def _known_caption(image: str) -> str | None:
    example = _matching_example(image)
    if example is None:
        return None
    path = example.with_suffix(".txt")
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return None


def _load_captioner():
    global _captioner
    if _captioner is not None:
        return _captioner
    import torch
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise gr.Error(
            "This transformers build has no Qwen2.5-VL. Type the description manually, "
            "or install transformers>=4.49 in the demo env."
        ) from exc
    print(f"[serve] loading caption model {CAPTION_MODEL}", flush=True)
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CAPTION_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval()
    _captioner = (model, processor)
    return _captioner


def _park_captioner() -> None:
    """Move the caption model off the GPU before the 8-GPU sampler starts."""
    if _captioner is None:
        return
    import torch
    model, _processor = _captioner
    if next(model.parameters()).device.type == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()


def _captioner_to_gpu() -> None:
    """Keep the caption model resident on cuda:0 between generations; moving
    16 GB of weights per upload is most of the caption latency."""
    import torch
    model, _processor = _load_captioner()
    if torch.cuda.is_available() and not _LOCK.locked():
        if next(model.parameters()).device.type != "cuda":
            model.to("cuda:0")


def _warm_captioner() -> None:
    try:
        with _CAPTION_LOCK:
            _captioner_to_gpu()
    except Exception as exc:
        print(f"[serve] caption model warm-up failed: {type(exc).__name__}: {exc}", flush=True)


def _run_caption(model, inputs, streamer, errors: list) -> None:
    import torch
    try:
        with torch.inference_mode():
            model.generate(**inputs, streamer=streamer, max_new_tokens=160, do_sample=False)
    except Exception as exc:
        errors.append(exc)
        streamer.end()


def describe_image(image: str | None):
    if not image:
        yield gr.update()
        return
    known = _known_caption(image)
    if known:
        yield known
        return
    from PIL import Image
    from transformers import TextIteratorStreamer
    text = ""
    errors: list = []
    with _CAPTION_LOCK:
        if _LOCK.locked():
            gr.Warning("All GPUs are busy generating a video; type the description or re-upload afterwards.")
            yield gr.update()
            return
        model, processor = _load_captioner()
        _captioner_to_gpu()
        device = next(model.parameters()).device
        picture = Image.open(image).convert("RGB")
        picture.thumbnail((768, 768))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": _CAPTION_INSTRUCTION},
            ],
        }]
        chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[chat], images=[picture], return_tensors="pt").to(device)
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        worker = threading.Thread(target=_run_caption, args=(model, inputs, streamer, errors), daemon=True)
        worker.start()
        for piece in streamer:
            text += piece
            yield " ".join(text.split())
        worker.join()
    if errors:
        raise gr.Error(f"Caption model failed: {type(errors[0]).__name__}: {errors[0]}")
    if not text.strip():
        raise gr.Error("The caption model returned an empty description. Type one manually.")


def preview_trajectory(image: str | None, trajectory: str, views: int) -> str | None:
    reference = _pose_file(image)
    if reference is None:
        return None
    try:
        poses = trajectory_for_preview(
            load_reference_poses(reference, int(views)), trajectory, int(views),
        )
        out_dir = ROOT / ".gradio_previews"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{trajectory}-{int(views)}-{uuid.uuid4().hex}.png"
        render_trajectory_preview(
            poses, out_path,
            f"{trajectory} camera poses · {reference.stem} · {len(poses)} views",
        )
        return str(out_path)
    except (OSError, ValueError, ImportError) as exc:
        print(f"[serve] trajectory preview failed: {exc}", flush=True)
        return None


def _free_gpu_mib() -> list[tuple[str, int]]:
    """Free memory of the GPUs torchrun will use, in CUDA_VISIBLE_DEVICES order."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True, encoding="utf-8", errors="replace",
    )
    by_index = {}
    for line in out.splitlines():
        if "," in line:
            index, mib = (part.strip() for part in line.split(",", 1))
            by_index[index] = int(mib)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    order = [item.strip() for item in visible.split(",") if item.strip()] if visible else sorted(by_index, key=int)
    return [(index, by_index[index]) for index in order if index in by_index]


def _require_free_gpus() -> None:
    """Refuse before torchrun. A full training job aborts NCCL with no Python traceback."""
    try:
        free = _free_gpu_mib()[:NGPU]
    except (OSError, subprocess.CalledProcessError, ValueError):
        return
    short = [(index, mib) for index, mib in free if mib < MIN_FREE_GIB * 1024]
    if len(free) < NGPU or short:
        detail = ", ".join(f"gpu{index}={mib / 1024:.1f} GiB free" for index, mib in free)
        raise gr.Error(
            f"Ulysses needs {NGPU} GPUs with at least {MIN_FREE_GIB:g} GiB free each. "
            f"This machine does not have that ({detail or 'no GPU'}). "
            "Start the demo on a node whose GPUs are not already training."
        )


def _pointcloud_preview(ply: Path, max_points: int = 400_000) -> Path | None:
    """Model3D renders .ply as a Gaussian splat, so show an RGB point GLB instead."""
    import numpy as np
    import trimesh
    with open(ply, "rb") as handle:
        count = 0
        while True:
            line = handle.readline()
            if not line:
                return None
            if line.startswith(b"element vertex"):
                count = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        data = np.fromfile(handle, sep=" ", dtype=np.float32)
    if count == 0 or data.size < count * 6:
        return None
    data = data[: count * 6].reshape(count, 6)
    if count > max_points:
        data = data[np.random.default_rng(0).choice(count, max_points, replace=False)]
    # The viewer frames the whole bounding box, so a few far points (sky)
    # would shrink the scene to a dot; keep the 1-99% core with a margin.
    lo, hi = np.percentile(data[:, :3], [1, 99], axis=0)
    margin = 0.1 * (hi - lo)
    core = np.all((data[:, :3] >= lo - margin) & (data[:, :3] <= hi + margin), axis=1)
    if core.sum() >= 1000:
        data = data[core]
    xyz = data[:, :3] - np.median(data[:, :3], axis=0)
    xyz /= max(float(np.abs(xyz).max()), 1e-6)
    xyz[:, 1:] *= -1  # OpenCV camera frame (y down, z forward) -> glTF (y up)
    colors = np.concatenate([data[:, 3:6], np.full((len(data), 1), 255, np.float32)], axis=1)
    out = ply.with_name(ply.stem + "_preview.glb")
    trimesh.PointCloud(xyz, colors=colors.astype(np.uint8)).export(out, file_type="glb")
    return out


def _web_video(src: Path | None) -> Path | None:
    """Browser copy of a sampler video. The sampler already writes browser-ready
    H.264 (crf 18, yuv420p, faststart), so it is served as is unless WEB_FPS > 0
    asks for motion interpolation, which softens frames and warps edges."""
    if src is None or WEB_FPS <= 0:
        return src
    out = src.with_name(src.stem + "_web.mp4")
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(src),
        "-vf", f"minterpolate=fps={WEB_FPS}:mi_mode=mci",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out),
    ]
    try:
        subprocess.run(command, check=True, timeout=120, capture_output=True)
        return out
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[serve] web video for {src.name} failed, serving original: {exc}", flush=True)
        return src


def _web_videos(*videos: Path | None) -> list[Path | None]:
    results: list[Path | None] = [None] * len(videos)

    def work(index: int) -> None:
        results[index] = _web_video(videos[index])

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(videos))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def _kill_group(proc: subprocess.Popen) -> None:
    """Stop generate.py, torchrun and every rank; they share one session."""
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _scale_rollout(rollout: dict, pose_scale: float) -> dict:
    """Scale the metric extent of a synthetic camera path in-place.

    ``gae_rollout`` stores the reference path's spatial extent as strings.
    ``synthesize_free_trajectory`` rescales the raw path to ``target_extents_xyz``
    (falling back to ``target_extent``), so multiplying those changes how far the
    injected camera travels for an uploaded image without altering the motion
    shape, view count, or direction. ``speed`` is left untouched because the
    extent rescale overrides it.
    """
    try:
        s = float(pose_scale)
    except (TypeError, ValueError):
        return rollout
    if not rollout.get("enabled") or s == 1.0 or s <= 0.0:
        return rollout
    te = rollout.get("target_extent")
    if te not in (None, ""):
        rollout["target_extent"] = f"{float(te) * s:.9g}"
    tex = rollout.get("target_extents_xyz")
    if tex not in (None, ""):
        rollout["target_extents_xyz"] = ",".join(
            f"{float(v) * s:.9g}" for v in str(tex).split(","))
    return rollout


def _scale_pose_frames(frames: list, pose_scale: float) -> bool:
    """Scale camera translations about frame 0 (rotations and intrinsics kept)."""
    import numpy as np

    s = float(pose_scale)
    if s == 1.0 or s <= 0.0 or not frames:
        return False
    c2w = np.asarray([f["c2w"] for f in frames], dtype=np.float64)
    c2w[:, :3, 3] = c2w[0, :3, 3] + s * (c2w[:, :3, 3] - c2w[0, :3, 3])
    for frame, pose in zip(frames, c2w):
        frame["c2w"] = pose.tolist()
    return True


def _prepare_job_scene(scene_dir: Path, image: str, prompt: str, trajectory: str,
                       views: int, pose_scale: float = 1.0):
    """Write the scene dir a worker job points at. Returns (frames, fps, mode)."""
    from scripts.demo import generate as gen
    pose_file = _pose_file(image)
    use_poses = trajectory == "example" and pose_file is not None
    motion = trajectory if trajectory != "example" else "forward"
    ns = argparse.Namespace(
        image=Path(image), prompt=prompt, resolution=RESOLUTION,
        num_views=views, total_views=views, fps=DEFAULT_FPS,
        trajectory=motion, speed=0.06, trajectory_reference_poses=pose_file,
        output=scene_dir.parent,
    )
    fps = DEFAULT_FPS
    if use_poses:
        fps = gen._load_gt_cameras(pose_file, views)[2] or DEFAULT_FPS
        rollout = {"enabled": False}
        bundled = _matching_example(image) is not None and pose_file != DEFAULT_POSES
        mode = "repository camera poses" if bundled else f"example camera poses ({pose_file.stem})"
    else:
        rollout = _scale_rollout(gen._rollout_meta(ns, views, motion), pose_scale)
        mode = f"synthetic {motion} trajectory"
    gen._prepare_scene(
        ns, pose_file if use_poses else None,
        scene_dir=scene_dir, scene_name=scene_dir.name,
        rollout=rollout, write_manifest=False, update_args=False,
    )
    meta_path = scene_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    if use_poses and _scale_pose_frames(meta["frames"], pose_scale):
        meta_path.write_text(json.dumps(meta, indent=2))
    if float(pose_scale) != 1.0:
        mode += f" · scale {float(pose_scale):.2g}x"
    return int(meta["num_frames"]), fps, mode


class _Worker:
    """One torchrun job that keeps the DiT, codec, DA3 and text encoder loaded.

    Requests are handed over as job files (see ``_ServeQueueScenes`` in
    eval_generation.py), so only the first request after a boot pays for
    loading weights and CUDA warm-up.
    """

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.views: int | None = None
        self.dir: Path | None = None
        self.seq = 0
        self.lock = threading.Lock()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, limit: int = 200_000) -> str:
        if self.dir is None or not (self.dir / "worker.log").is_file():
            return ""
        with open(self.dir / "worker.log", "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - limit))
            return handle.read().decode("utf-8", errors="replace")

    def stop(self) -> None:
        if self.proc is not None:
            if self.alive():
                try:
                    self._submit({"stop": True})
                    self.proc.wait(timeout=20)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            _kill_group(self.proc)
        self.proc = None
        self.views = None
        if self.dir is not None:
            shutil.rmtree(self.dir / "rank_tmp", ignore_errors=True)

    def _submit(self, job: dict) -> int:
        seq = self.seq
        path = self.dir / f"job_{seq:06d}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job))
        os.replace(tmp, path)
        self.seq += 1
        return seq

    def _boot(self, views: int) -> None:
        self.stop()
        _require_free_gpus()
        WORKER_ROOT.mkdir(parents=True, exist_ok=True)
        for old in WORKER_ROOT.glob("w-*"):
            shutil.rmtree(old, ignore_errors=True)
        self.dir = WORKER_ROOT / f"w-{uuid.uuid4().hex[:12]}"
        (self.dir / "rank_tmp").mkdir(parents=True)
        command = [
            sys.executable, str(ROOT / "scripts" / "demo" / "generate.py"),
            "--image", str(EXAMPLE_IMAGE),
            "--prompt-file", str(EXAMPLE_IMAGE.with_suffix(".txt")),
            "--hf-repo", HF_REPO,
            "--ckpt-dir", str(CKPT_DIR),
            "--output", str(self.dir / "boot"),
            "--num-views", str(views),
            "--total-views", str(views),
            "--sample-steps", "2",
            "--seed", "0",
            "--ulysses-size", str(NGPU),
            # FREE_ROLLOUT at launch selects the chunked path that handles both
            # repository poses and synthetic trajectories per scene.
            "--free-rollout", "--trajectory", "forward",
            "--save-pointcloud",
            "--pc-stride", str(PC_STRIDE),
            "--serve-queue", str(self.dir),
        ]
        if CFG_PARALLEL:
            command.append("--cfg-parallel")
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts' / 'eval'}:{env.get('PYTHONPATH', '')}"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Ranks other than 0 write duplicate outputs under TMPDIR.
        env["TMPDIR"] = str(self.dir / "rank_tmp")
        print(f"[serve] booting worker ({views} views, cfg_parallel={CFG_PARALLEL}): {self.dir}", flush=True)
        with open(self.dir / "worker.log", "ab") as log:
            self.proc = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.seq = 0
        self.views = views
        started = time.time()
        self.run(
            EXAMPLE_IMAGE, EXAMPLE_IMAGE.with_suffix(".txt").read_text(encoding="utf-8").strip(),
            "example", views, steps=2, cfg_scale=2.0, seed=0,
            save_pointcloud=True, filter_pointcloud=True,
            out_dir=self.dir / "warmup", timeout=BOOT_TIMEOUT,
        )
        print(f"[serve] worker ready in {time.time() - started:.0f}s", flush=True)

    def ensure(self, views: int) -> bool:
        """Boot when needed; returns True when this call booted the worker."""
        if self.alive() and self.views == views:
            return False
        self._boot(views)
        return True

    def run(self, image, prompt, trajectory, views, *, steps, cfg_scale, seed,
            save_pointcloud, filter_pointcloud, pose_scale: float = 1.0,
            out_dir: Path, timeout: int) -> str:
        scene_dir = out_dir / "_scene"
        frames, fps, mode = _prepare_job_scene(
            scene_dir, str(image), prompt, trajectory, views, pose_scale)
        seq = self._submit({
            "scene_name": scene_dir.name,
            "scene_dir": str(scene_dir.resolve()),
            "num_frames": frames,
            "prompt": prompt,
            "sample_steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "seed": int(seed),
            "video_fps": int(fps),
            "save_pointcloud": bool(save_pointcloud),
            "pc_filter": bool(save_pointcloud and filter_pointcloud),
            "output": str(out_dir.resolve()),
        })
        done = self.dir / f"done_{seq:06d}.json"
        deadline = time.time() + timeout
        while not done.is_file():
            if not self.alive():
                code = self.proc.returncode if self.proc is not None else None
                log = self.log_tail()
                (out_dir / "serve.log").write_text(log, encoding="utf-8")
                self.stop()
                raise gr.Error(f"The GPU worker exited (code {code}).\n\n{_failure_excerpt(log)}")
            if time.time() > deadline:
                log = self.log_tail()
                (out_dir / "serve.log").write_text(log, encoding="utf-8")
                self.stop()
                raise gr.Error(f"Generation timed out after {timeout}s; the worker was restarted.\n\n{log[-3000:]}")
            time.sleep(0.2)
        result = json.loads(done.read_text())
        if not result.get("ok"):
            print(f"[serve] job {seq} post-processing warning: {result.get('error')}", flush=True)
        return mode


_WORKER = _Worker()
atexit.register(_WORKER.stop)


def _boot_worker_in_background() -> None:
    def boot() -> None:
        with _WORKER.lock:
            try:
                _WORKER.ensure(max(VIEW_CHOICES))
            except Exception as exc:
                print(f"[serve] worker boot failed (will retry on first request): {exc}", flush=True)
    threading.Thread(target=boot, daemon=True).start()


def _make_room_on_gpu0() -> None:
    """Keep the caption model resident unless rank 0 needs its memory."""
    with _CAPTION_LOCK:
        if _captioner is None:
            return
        try:
            free_gib = _free_gpu_mib()[0][1] / 1024
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
            free_gib = 0.0
        if free_gib < MIN_FREE_GIB:
            _park_captioner()


def _failure_excerpt(log: str) -> str:
    marker = "Traceback (most recent call last):"
    chunks = []
    start = 0
    while len(chunks) < 2:
        index = log.find(marker, start)
        if index < 0:
            break
        chunks.append(log[index:index + 1800])
        start = index + len(marker)
    text = "\n\n".join(chunks) if chunks else log[-4000:]
    return text[-4000:]


def generate_i2v(
    image: str | None,
    prompt: str,
    trajectory: str,
    views: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    save_pointcloud: bool,
    filter_pointcloud: bool = True,
    pose_scale: float = 1.0,
):
    if not image:
        raise gr.Error("Upload an image first.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please provide a scene description.")
    if not _LOCK.acquire(blocking=False):
        raise gr.Error("Another video is generating. Wait for it to finish.")
    try:
        _make_room_on_gpu0()
        yield from _generate_locked(
            image, prompt, trajectory, views, steps, cfg_scale, seed, save_pointcloud,
            filter_pointcloud, pose_scale,
        )
    finally:
        _LOCK.release()
        threading.Thread(target=_warm_captioner, daemon=True).start()


def _generate_locked(image, prompt, trajectory, views, steps, cfg_scale, seed,
                     save_pointcloud, filter_pointcloud, pose_scale=1.0):
    views, steps = int(views), int(steps)
    seed = 42 if seed is None else int(seed)
    run_dir = OUTPUT_ROOT / f"i2v-{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with _WORKER.lock:
        booted = _WORKER.ensure(views)
        mode = _WORKER.run(
            image, prompt, trajectory, views,
            steps=steps, cfg_scale=cfg_scale, seed=seed,
            save_pointcloud=save_pointcloud, filter_pointcloud=filter_pointcloud,
            pose_scale=pose_scale, out_dir=run_dir, timeout=TIMEOUT,
        )
    video = _first(run_dir, "_pred.mp4")
    if video is None:
        log = _WORKER.log_tail()
        (run_dir / "serve.log").write_text(log, encoding="utf-8")
        raise gr.Error(f"Generation finished but no RGB video (_pred.mp4) was written.\n\n{_failure_excerpt(log)}")
    depth = _first(run_dir, "_depth.mp4")
    cloud = _first(run_dir, "_pred_pointcloud.ply") if save_pointcloud else None
    summary = (
        f"GAE-64 · Ulysses {NGPU} GPU{' · CFG parallel' if CFG_PARALLEL else ''} · {views} views · "
        f"{steps} steps · seed {seed} · {mode}"
    )
    loaded = " (includes loading the models)" if booted else ""
    missing = [name for name, path in (("depth video", depth), ("point cloud", cloud if save_pointcloud else True)) if path is None]
    if missing:
        (run_dir / "serve.log").write_text(_WORKER.log_tail(), encoding="utf-8")
        loaded += f" · no {' / '.join(missing)} was written; see {run_dir / 'serve.log'}"

    cloud_view: list[Path | None] = [None]

    def build_cloud() -> None:
        if cloud is None:
            return
        try:
            cloud_view[0] = _pointcloud_preview(cloud)
        except Exception as exc:
            print(f"[serve] point cloud preview failed: {type(exc).__name__}: {exc}", flush=True)

    cloud_thread = threading.Thread(target=build_cloud)
    cloud_thread.start()
    web_video, web_depth = _web_videos(video, depth)
    cloud_thread.join()
    yield (
        str(web_video),
        str(web_depth) if web_depth else None,
        str(cloud_view[0]) if cloud_view[0] else None,
        str(cloud) if cloud else None,
        f"{summary} · {time.time() - started:.0f}s{loaded}",
    )


def build() -> gr.Blocks:
    with gr.Blocks(title="GAE Ulysses I2V") as demo:
        gr.Markdown(
            f"""
# GAE — camera-controlled video

Upload an image. A scene description is filled in automatically and can be edited.
Generation uses **{NGPU} GPUs** with Ulysses sequence parallel. The models stay loaded
between requests, so a video takes about a minute at 20 steps. One video runs at a time;
changing the view count reloads the models once.
            """
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### Choices")
                trajectory = gr.Dropdown(label="Camera trajectory", choices=TRAJECTORIES, value="example")
                views = gr.Dropdown(label="Views", choices=VIEW_CHOICES, value=81)
                pose_scale = gr.Slider(
                    label="Camera motion scale",
                    minimum=0.25, maximum=3.0, step=0.05, value=1.0,
                    info="Scales how far the camera travels, for example poses and "
                         "synthetic trajectories alike. 1.0 = repository scale.",
                )
                steps = gr.Slider(label="Sampling steps", minimum=10, maximum=50, step=5, value=20)
                cfg = gr.Slider(label="CFG scale", minimum=1.0, maximum=4.0, step=0.1, value=2.0)
                seed = gr.Number(label="Seed", value=42, precision=0)
                pointcloud = gr.Checkbox(label="Also write a point cloud (adds ~10 s)", value=True)
                pc_filter = gr.Checkbox(label="Filter depth-edge and noisy points", value=True)
            with gr.Column(scale=2):
                image = gr.Image(label="Input image", type="filepath", sources=["upload", "clipboard"], height=300)
                prompt = gr.Textbox(
                    label="Scene description", lines=4,
                    placeholder="Filled automatically after you upload an image. You can edit it.",
                )
                preview = gr.Image(label="Camera path preview", type="filepath", interactive=False, height=240)
                run = gr.Button("Generate video", variant="primary")
            with gr.Column(scale=2):
                video = gr.Video(label="Generated RGB", autoplay=True, loop=True, height=300)
                depth = gr.Video(label="Decoded depth", autoplay=True, loop=True, height=300)
                cloud_view = gr.Model3D(label="Point cloud", clear_color=[1.0, 1.0, 1.0, 1.0], height=360)
                cloud = gr.File(label="Point cloud (.ply)")
                status = gr.Markdown()
        examples = _examples()
        if examples:
            gr.Examples(examples=examples, inputs=[image, prompt, trajectory], examples_per_page=8)
        image.change(describe_image, inputs=image, outputs=prompt)
        preview_inputs = [image, trajectory, views]
        for event in (image.change, trajectory.change, views.change):
            event(preview_trajectory, inputs=preview_inputs, outputs=preview)
        run.click(
            generate_i2v,
            inputs=[image, prompt, trajectory, views, steps, cfg, seed, pointcloud,
                    pc_filter, pose_scale],
            outputs=[video, depth, cloud_view, cloud, status],
            concurrency_limit=1,
        )
    return demo


def main(host: str | None = None, port: int | None = None,
         share: bool | None = None) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    threading.Thread(target=_warm_captioner, daemon=True).start()
    _boot_worker_in_background()
    demo = build()
    if hasattr(demo, "queue"):
        demo.queue(max_size=8)
    auth = os.environ.get("GRADIO_AUTH", "")
    auth_arg = tuple(auth.split(":", 1)) if ":" in auth else None
    if share is None:
        share = os.environ.get("GRADIO_SHARE", "1") == "1"
    if port is None:
        port = int(os.environ.get("PORT", "7860"))
    host = host or os.environ.get("HOST", "0.0.0.0")
    allowed = [str(OUTPUT_ROOT), str(ROOT / ".gradio_previews"), str(ROOT / "examples")]
    print(f"[serve] ulysses_size={NGPU} host={host} port={port} share={share} output={OUTPUT_ROOT}", flush=True)
    print(f"[serve] caption model={CAPTION_MODEL}", flush=True)
    try:
        demo.launch(
            server_name=host,
            server_port=port,
            share=share,
            auth=auth_arg,
            allowed_paths=allowed,
        )
    except Exception as exc:
        if not share:
            raise
        print(f"[serve] public share link failed ({exc}); binding local port only", flush=True)
        demo.launch(
            server_name=host,
            server_port=port,
            share=False,
            auth=auth_arg,
            allowed_paths=allowed,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAE Ulysses I2V web demo")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"),
                        help="Bind address; 0.0.0.0 accepts remote connections.")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", "7860")),
                        help="Port to listen on.")
    parser.add_argument("--share", dest="share", action="store_true", default=None,
                        help="Also open a Gradio public link.")
    parser.add_argument("--no-share", dest="share", action="store_false",
                        help="Never open a Gradio public link.")
    _args = parser.parse_args()
    main(host=_args.host, port=_args.port, share=_args.share)
