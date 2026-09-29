#!/usr/bin/env python3
"""Generate a camera-controlled video and point cloud from one image + prompt.

This is the public single-scene wrapper around ``eval_generation.py``. It
creates the small scene manifest expected by the research evaluator, then runs
the released GAE flow sampler.

Example:
    python scripts/demo/generate.py \
        --image examples/scenes/forest_lake_trail.jpg \
        --prompt-file examples/scenes/forest_lake_trail.txt \
        --hf-repo TencentARC/GAE-D64-1B \
        --output results/forest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.demo.trajectory_utils import (
    load_reference_poses, reference_forward_sign, trajectory_extent,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--prompt", default=None, help="Prompt text; or use --prompt-file.")
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help="Read the prompt from a text file, e.g. examples/scenes/<scene>.txt.")
    parser.add_argument(
        "--batch-json", type=Path, default=None,
        help="JSON list of {image, prompt or prompt_file, output, trajectory}. "
             "One process loads the weights and runs every job.",
    )
    parser.add_argument(
        "--hf-repo", default=None,
        help="Hugging Face repo id (default env GAE_HF_REPO or TencentARC/GAE-D64-1B). "
             "Downloads codec + flow into --ckpt-dir when local files are missing.")
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "ckpts")
    parser.add_argument("--flow-ckpt", type=Path, default=None)
    parser.add_argument("--codec-ckpt", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/flow_gae64.yaml")
    parser.add_argument("--codec-config", type=Path, default=ROOT / "configs/gae_64.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-views", type=int, default=81,
                        help="views per denoising chunk; 81 matches the shipped "
                             "flow recipe (V=81) and needs no rollout")
    parser.add_argument("--total-views", type=int, default=81,
                        help="total output frames; exceeding --num-views enables "
                             "chunked rollout")
    parser.add_argument("--roll-cond-num", type=int, default=1,
                        help="frames carried over between rollout chunks")
    parser.add_argument("--resolution", type=int, nargs=2, default=(378, 672),
                        metavar=("HEIGHT", "WIDTH"),
                        help="H W; default 378x672 matches the shipped 672x378 flow recipe")
    parser.add_argument(
        "--poses", type=Path, default=None,
        help="GT camera npz (c2w Nx4x4, K Nx3x3). Default: <image_stem>_poses.npz "
             "next to --image when that file exists.")
    parser.add_argument(
        "--free-rollout", action="store_true",
        help="Ignore GT poses and synthesize a camera path (--trajectory / --speed).")
    parser.add_argument(
        "--trajectory", choices=("forward", "backward", "turn_left", "turn_right"),
        default="forward",
    )
    parser.add_argument("--speed", type=float, default=0.06)
    parser.add_argument(
        "--trajectory-reference-poses", type=Path, default=None,
        help="Pose NPZ whose prefix path length defines the scale of synthetic trajectories.",
    )
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument(
        "--ulysses-size", type=int, default=1,
        help="GPUs for one clip. 1 is single-GPU. 8 relaunches under torchrun "
             "and shards the DiT sequence with Ulysses (GAE-D64 has 16 heads).",
    )
    parser.add_argument(
        "--cfg-parallel", action="store_true",
        help="Split --ulysses-size in half and run the CFG conditional and "
             "unconditional branches at the same time.",
    )
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--save-pointcloud", action="store_true", default=True,
                        help="Write DPT depth + .ply (default on).")
    parser.add_argument("--no-pointcloud", dest="save_pointcloud", action="store_false")
    parser.add_argument("--pc-stride", type=int, default=4)
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if args.batch_json is not None:
        if not args.batch_json.is_file():
            raise SystemExit(f"--batch-json does not exist: {args.batch_json}")
        jobs = json.loads(args.batch_json.read_text())
        if not isinstance(jobs, list) or not jobs:
            raise SystemExit("--batch-json must be a non-empty list")
        args.jobs = jobs
        args.prompt = "batch"
        if args.image is None:
            args.image = Path(jobs[0]["image"])
    else:
        args.jobs = None
        if args.image is None:
            raise SystemExit("Provide --image, or --batch-json for several scenes.")
        if args.prompt_file is not None:
            if not args.prompt_file.is_file():
                raise SystemExit(f"--prompt-file does not exist: {args.prompt_file}")
            args.prompt = args.prompt_file.read_text().strip()
        if not args.prompt:
            raise SystemExit("Provide a prompt via --prompt or --prompt-file.")
    if args.hf_repo or args.flow_ckpt is None or args.codec_ckpt is None:
        from gae.hub import DEFAULT_REPO, download_weights, extract_da3_stats
        repo = args.hf_repo or os.environ.get("GAE_HF_REPO") or DEFAULT_REPO
        paths = download_weights(repo, size="64", out_dir=args.ckpt_dir)
        tar = paths.get("da3_stats_giant_5ds.tar")
        if tar is not None:
            extract_da3_stats(tar, ROOT / "model_stats" / "da3_giant_5ds")
        if args.codec_ckpt is None:
            args.codec_ckpt = paths["gae_64.pt"]
        if args.flow_ckpt is None:
            args.flow_ckpt = paths["flow_gae64.pt"]
    return args, extra


def _check_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} does not exist: {path}")


def _resolve_poses_path(args: argparse.Namespace) -> Path | None:
    if args.free_rollout or "--trajectory" in sys.argv:
        return None
    if args.poses is not None:
        return args.poses
    sibling = args.image.with_name(f"{args.image.stem}_poses.npz")
    return sibling if sibling.is_file() else None


def _resolve_trajectory_reference_path(args: argparse.Namespace) -> Path | None:
    """Find the example pose path used to metric-normalize free rollouts."""
    candidates = []
    if args.trajectory_reference_poses is not None:
        candidates.append(args.trajectory_reference_poses)
    candidates.append(args.image.with_name(f"{args.image.stem}_poses.npz"))
    candidates.append(ROOT / "examples" / "scenes" / f"{args.image.stem}_poses.npz")
    # Uploaded images have no sibling pose file; use the shipped canonical
    # example path so all trajectory choices retain repository-scale motion.
    candidates.append(ROOT / "examples" / "scenes" / "forest_lake_trail_poses.npz")
    for path in candidates:
        if path is not None and path.is_file():
            return path
    return None


def _load_gt_cameras(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, int | None]:
    data = np.load(path)
    if "c2w" not in data or "K" not in data:
        raise SystemExit(f"{path} must contain 'c2w' (Nx4x4) and 'K' (Nx3x3)")
    c2w = np.asarray(data["c2w"], dtype=np.float32)
    K = np.asarray(data["K"], dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise SystemExit(f"{path}: c2w expected (N,4,4), got {c2w.shape}")
    if K.ndim != 3 or K.shape[1:] != (3, 3):
        raise SystemExit(f"{path}: K expected (N,3,3), got {K.shape}")
    n_cam = min(int(n), int(c2w.shape[0]), int(K.shape[0]))
    fps = int(data["fps"]) if "fps" in data else None
    return c2w[:n_cam], K[:n_cam], fps


def _read_prompt(job: dict) -> str:
    if job.get("prompt_file"):
        path = Path(job["prompt_file"])
        if not path.is_file():
            raise SystemExit(f"prompt file does not exist: {path}")
        return path.read_text().strip()
    prompt = str(job.get("prompt") or "").strip()
    if not prompt:
        raise SystemExit(f"job {job.get('image')!r} needs prompt or prompt_file")
    return prompt


def _rollout_meta(args: argparse.Namespace, total_views: int, trajectory: str) -> dict:
    meta = {
        "enabled": True,
        "motion": trajectory,
        "views": int(total_views),
        "speed": float(args.speed),
    }
    reference = _resolve_trajectory_reference_path(args)
    if reference is None:
        return meta
    try:
        ref = load_reference_poses(reference, total_views)
        target = trajectory_extent(ref)
    except (OSError, ValueError) as exc:
        print(f"[generate] warning: cannot read trajectory reference {reference}: {exc}", flush=True)
        return meta
    if target > 0.0:
        meta["target_extent"] = f"{target:.9g}"
        meta["target_extents_xyz"] = ",".join(f"{float(v):.9g}" for v in np.ptp(ref[:, :3, 3], axis=0))
        meta["fwd_sign"] = str(reference_forward_sign(ref))
        meta["target_direction"] = ",".join(
            f"{float(v):.9g}" for v in (ref[-1, :3, 3] - ref[0, :3, 3])
        )
        print(
            f"[generate] synthetic trajectory diameter: {target:.4f}m "
            f"from {reference} ({len(ref)} views)",
            flush=True,
        )
    return meta


def _write_still_video(path: Path, image: np.ndarray, n: int, fps: float) -> None:
    """Repeat the conditioning image losslessly. Frame 0 is the model's
    conditioning view, and mp4v leaves visible blocking on detailed photos
    (about 30 dB PSNR on forest_lake_trail)."""
    height, width = image.shape[:2]
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:g}",
        "-i", "-", "-c:v", "libx264rgb", "-qp", "0", "-preset", "ultrafast", str(path),
    ]
    try:
        subprocess.run(command, input=np.ascontiguousarray(image).tobytes() * n,
                       check=True, capture_output=True)
        return
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[generate] lossless input video failed ({exc}); falling back to mp4v", flush=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("OpenCV cannot create the temporary input video")
    for _ in range(n):
        writer.write(image)
    writer.release()


def _prepare_scene(
    args: argparse.Namespace,
    poses_path: Path | None,
    *,
    scene_dir: Path | None = None,
    scene_name: str | None = None,
    rollout: dict | None = None,
    write_manifest: bool = True,
    update_args: bool = True,
) -> Path:
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    height, width = args.resolution
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    n = args.num_views
    if poses_path is not None:
        _check_file(poses_path, "poses")
        c2w, Ks, fps = _load_gt_cameras(poses_path, n)
        n = int(c2w.shape[0])
        if update_args:
            args.num_views = n
            args.total_views = min(int(args.total_views), n)
            if fps:
                args.fps = fps
        frames = [
            {
                "index": i,
                "name": f"{i:06d}",
                "c2w": c2w[i].tolist(),
                "K": Ks[i].tolist(),
            }
            for i in range(n)
        ]
        print(f"[generate] using GT cameras from {poses_path} ({n} frames)", flush=True)
    else:
        focal = 0.8 * max(height, width)
        K = [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]]
        frames = [
            {
                "index": i,
                "name": f"{i:06d}",
                "c2w": np.eye(4, dtype=np.float32).tolist(),
                "K": K,
            }
            for i in range(n)
        ]
        label = (rollout or {}).get("motion") or args.trajectory
        print(f"[generate] no GT poses; synthetic '{label}' trajectory", flush=True)

    scene = scene_dir or (args.output / "_input_scene")
    name = scene_name or args.image.stem
    scene.mkdir(parents=True, exist_ok=True)
    _write_still_video(scene / "video.mp4", image, n, float(args.fps))
    meta = {
        "scene_id": name,
        "num_frames": n,
        "height": height,
        "width": width,
        "rgb_resolution": [height, width],
        "caption": args.prompt,
        "frames": frames,
    }
    if rollout is not None:
        meta["gae_rollout"] = rollout
    (scene / "meta.json").write_text(json.dumps(meta, indent=2))
    (scene / "caption.txt").write_text(args.prompt.strip() + "\n")
    if not write_manifest:
        return scene

    manifest = args.output / "_input_manifest.json"
    manifest.write_text(json.dumps({
        "dataset": "scannetpp",
        "num_views": n,
        "scenes": [{
            "scene_name": name,
            "scene_dir": str(scene.resolve()),
            "img_names": [f"{i:06d}" for i in range(n)],
            "ds_type": "scannetpp",
        }],
    }, indent=2))
    return manifest


def _launch_eval(args: argparse.Namespace, extra: list[str], manifest: Path, env: dict, num_scenes: int) -> int:
    command = [
        sys.executable, str(ROOT / "scripts/eval/eval_generation.py"),
        "--dit-ckpt", str(args.flow_ckpt),
        "--vae-ckpt", str(args.codec_ckpt),
        "--gld-config", str(args.codec_config),
        "--config", str(args.config),
        "--scene-manifest", str(manifest),
        "--dataset", "scannetpp",
        "--mode", "generate",
        "--prompt", args.prompt,
        "--cond-num", "1",
        "--num-scenes", str(num_scenes),
        "--num-views", str(args.num_views),
        "--total-views", str(args.total_views),
        "--roll-cond-num", str(args.roll_cond_num),
        "--resolution", str(args.resolution[0]), str(args.resolution[1]),
        "--sample-steps", str(args.sample_steps),
        "--cfg-scale", str(args.cfg_scale),
        "--seed", str(args.seed),
        "--video-fps", str(args.fps),
        "--pc-stride", str(args.pc_stride),
        "--output-dir", str(args.output),
        *extra,
    ]
    command.append("--save-pointcloud" if args.save_pointcloud else "--no-pointcloud")
    if args.ulysses_size < 1:
        raise SystemExit("--ulysses-size must be >= 1")
    if args.cfg_parallel and args.ulysses_size < 2:
        raise SystemExit("--cfg-parallel needs --ulysses-size >= 2")
    if args.cfg_parallel and args.ulysses_size % 2 != 0:
        raise SystemExit("--cfg-parallel needs an even --ulysses-size")
    if args.ulysses_size > 1:
        command = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone",
            "--nproc_per_node", str(args.ulysses_size),
            *command[1:],
            "--ulysses",
        ]
        if args.cfg_parallel:
            command.append("--cfg-parallel")
    print("[generate]", " ".join(command), flush=True)
    print(f"[generate] jobs={num_scenes} (weights load once)", flush=True)
    return subprocess.call(command, cwd=ROOT, env=env)


def _apply_rollout_env(env: dict, rollout: dict) -> None:
    env["FREE_ROLLOUT"] = "1"
    env["FREE_ROLLOUT_VIEWS"] = str(rollout["views"])
    env["FREE_ROLLOUT_MOTION"] = str(rollout["motion"])
    env["FREE_ROLLOUT_SPEED"] = str(rollout["speed"])
    for src, key in (
        ("target_extent", "FREE_ROLLOUT_TARGET_EXTENT"),
        ("target_extents_xyz", "FREE_ROLLOUT_TARGET_EXTENTS_XYZ"),
        ("fwd_sign", "FREE_ROLLOUT_FWD_SIGN"),
        ("target_direction", "FREE_ROLLOUT_TARGET_DIRECTION"),
    ):
        if rollout.get(src) not in (None, ""):
            env[key] = str(rollout[src])


def _run_batch(args: argparse.Namespace, extra: list[str]) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    index = []
    first_rollout = None
    saved_image, saved_prompt, saved_trajectory = args.image, args.prompt, args.trajectory
    for i, job in enumerate(args.jobs):
        if "image" not in job or "output" not in job:
            raise SystemExit("each batch job needs image and output")
        args.image = Path(job["image"])
        _check_file(args.image, "image")
        args.prompt = _read_prompt(job)
        trajectory = str(job.get("trajectory") or "").strip()
        if trajectory and trajectory not in ("forward", "backward", "turn_left", "turn_right"):
            raise SystemExit(f"unknown trajectory: {trajectory}")
        if trajectory:
            poses_path = None
            args.trajectory = trajectory
            rollout = _rollout_meta(args, args.total_views, trajectory)
        else:
            poses_raw = job.get("poses")
            if poses_raw:
                poses_path = Path(poses_raw)
            else:
                sibling = args.image.with_name(f"{args.image.stem}_poses.npz")
                poses_path = sibling if sibling.is_file() else None
            if poses_path is None:
                args.trajectory = "forward"
                rollout = _rollout_meta(args, args.total_views, "forward")
            else:
                rollout = {"enabled": False}
        if rollout.get("enabled") and first_rollout is None:
            first_rollout = rollout
        scene_name = f"{args.image.stem}__{trajectory or 'gt'}"
        scene_dir = args.output / "_scenes" / f"{i:03d}_{scene_name}"
        _prepare_scene(
            args, poses_path,
            scene_dir=scene_dir, scene_name=scene_name, rollout=rollout,
            write_manifest=False, update_args=False,
        )
        n = int(json.loads((scene_dir / "meta.json").read_text())["num_frames"])
        entries.append({
            "scene_name": scene_name,
            "scene_dir": str(scene_dir.resolve()),
            "img_names": [f"{k:06d}" for k in range(n)],
            "ds_type": "scannetpp",
        })
        index.append({
            "output": str(job["output"]),
            "scene": args.image.stem,
            "trajectory": trajectory,
        })
        print(f"[generate] queued {i + 1}/{len(args.jobs)} {scene_name}", flush=True)
    args.image, args.prompt, args.trajectory = saved_image, saved_prompt, saved_trajectory
    if entries and any(int(json.loads(Path(e["scene_dir"]).joinpath("meta.json").read_text())["num_frames"]) != args.num_views for e in entries):
        print("[generate] warning: a job wrote a different frame count than --num-views", flush=True)
    manifest = args.output / "_input_manifest.json"
    manifest.write_text(json.dumps({
        "dataset": "scannetpp",
        "num_views": args.num_views,
        "scenes": entries,
    }, indent=2))
    (args.output / "job_index.json").write_text(json.dumps(index, indent=2))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts' / 'eval'}:{env.get('PYTHONPATH', '')}"
    env.pop("FREE_ROLLOUT", None)
    if first_rollout is not None:
        _apply_rollout_env(env, first_rollout)
    return _launch_eval(args, extra, manifest, env, len(entries))


def _resolve_latent_stats(args: argparse.Namespace) -> None:
    """Point the flow config's ``latent_stats`` at a file that exists.

    The shipped config names ``ckpts/latent_stats_gae_64.pt`` relative to the
    checkout, but weights may live in another ``--ckpt-dir``. eval_generation
    silently skips latent (de)normalization when the file is missing, which
    turns the output into noise, so rewrite the path to the downloaded copy.
    """
    text = args.config.read_text()
    match = re.search(r"^latent_stats:.*$", text, flags=re.M)
    if match is None:
        return
    value = Path(re.split(r"\s+#", match.group(0).split(":", 1)[1])[0].strip().strip("'\""))
    if (value if value.is_absolute() else ROOT / value).is_file():
        return
    candidate = (args.ckpt_dir / value.name).resolve()
    if not candidate.is_file():
        raise SystemExit(
            f"latent stats not found: {value} (also tried {candidate}); "
            "run scripts/demo/download_checkpoints.py or pass --ckpt-dir"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    patched = args.output / f"_{args.config.stem}.yaml"
    patched.write_text(text[:match.start()] + f'latent_stats: "{candidate}"' + text[match.end():])
    print(f"[generate] latent stats: {candidate}", flush=True)
    args.config = patched


def main() -> int:
    args, extra = parse_args()
    for path, label in (
        (args.flow_ckpt, "flow checkpoint"),
        (args.codec_ckpt, "codec checkpoint"),
        (args.config, "flow config"),
        (args.codec_config, "codec config"),
    ):
        _check_file(path, label)
    _resolve_latent_stats(args)
    if args.total_views > args.num_views and not 0 < args.roll_cond_num < args.num_views:
        raise SystemExit("--roll-cond-num must be in (0, num-views) for long rollout")
    if args.jobs:
        return _run_batch(args, extra)

    _check_file(args.image, "image")
    args.output.mkdir(parents=True, exist_ok=True)
    poses_path = _resolve_poses_path(args)
    manifest = _prepare_scene(args, poses_path)
    trajectory_reference_path = (
        _resolve_trajectory_reference_path(args) if poses_path is None else None
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts' / 'eval'}:{env.get('PYTHONPATH', '')}"
    env.pop("FREE_ROLLOUT", None)
    if poses_path is None:
        env.update({
            "FREE_ROLLOUT": "1",
            "FREE_ROLLOUT_VIEWS": str(args.total_views),
            "FREE_ROLLOUT_MOTION": args.trajectory,
            "FREE_ROLLOUT_SPEED": str(args.speed),
        })
        if trajectory_reference_path is not None:
            try:
                ref = load_reference_poses(trajectory_reference_path, args.total_views)
                target = trajectory_extent(ref)
            except (OSError, ValueError) as exc:
                print(f"[generate] warning: cannot read trajectory reference {trajectory_reference_path}: {exc}", flush=True)
                target = 0.0
            if target > 0.0:
                env["FREE_ROLLOUT_TARGET_EXTENT"] = str(target)
                env["FREE_ROLLOUT_TARGET_EXTENTS_XYZ"] = ",".join(
                    f"{float(v):.9g}" for v in np.ptp(ref[:, :3, 3], axis=0)
                )
                env["FREE_ROLLOUT_FWD_SIGN"] = str(reference_forward_sign(ref))
                env["FREE_ROLLOUT_TARGET_DIRECTION"] = ",".join(
                    f"{float(v):.9g}" for v in (ref[-1, :3, 3] - ref[0, :3, 3])
                )
                print(
                    f"[generate] synthetic trajectory diameter: {target:.4f}m "
                    f"from {trajectory_reference_path} ({len(ref)} views)",
                    flush=True,
                )
    return _launch_eval(args, extra, manifest, env, 1)


if __name__ == "__main__":
    raise SystemExit(main())
