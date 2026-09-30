"""Run the released GAE codec reconstruction and export RGB/depth/point cloud."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import imageio.v3 as iio

from gae import GAE

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "scripts" / "eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _depth_to_color(
    depth: torch.Tensor,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    """Render depth with the project's viridis visualization convention."""
    import matplotlib.cm as cm

    arr = depth.detach().float().cpu().squeeze().numpy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros((*arr.shape, 3), dtype=np.uint8)
    if lo is None or hi is None:
        lo, hi = float(arr[finite].min()), float(arr[finite].max())
    if hi - lo > 1e-6:
        norm = (arr - lo) / (hi - lo)
    else:
        norm = np.zeros_like(arr)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    colored = cm.viridis(np.clip(norm, 0.0, 1.0))[..., :3]
    return (colored * 255.0 + 0.5).astype(np.uint8)


def _read_input(path: Path, size: tuple[int, int], max_frames: int = 0) -> tuple[np.ndarray, float]:
    h, w = size
    if path.suffix.lower() in {".mp4", ".mov", ".webm", ".avi", ".mkv"}:
        frames = iio.imread(path, index=None)
        fps = float(iio.immeta(path).get("fps", 8.0))
    else:
        frames = np.asarray(Image.open(path).convert("RGB"))[None]
        fps = 1.0
    if frames.ndim != 4:
        raise ValueError(f"expected RGB video frames [V,H,W,3], got {frames.shape}")
    if max_frames > 0 and len(frames) > max_frames:
        print(f"{path}: using the first {max_frames} of {len(frames)} frames", flush=True)
        frames = frames[:max_frames]
    resized = []
    for frame in frames:
        resized.append(np.asarray(Image.fromarray(frame).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)))
    return np.stack(resized), fps


def _save_video(frames: np.ndarray, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.asarray(frames, dtype=np.uint8), fps=fps, codec="libx264")


def _views_first(tensor: torch.Tensor, channels: int | None) -> torch.Tensor:
    """[V,C,H,W] (``channels`` set) or [V,H,W] from the codec's batched layouts."""
    if tensor.ndim == 5 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if channels is None:
        # The released pipeline returns [1,V,H,W]; compatible backbones may
        # return [V,1,H,W] or [V,H,W].
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim == 4 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        expected = 3
    else:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        expected = 4
    if tensor.ndim != expected:
        raise RuntimeError(f"unexpected reconstruction shape: {tuple(tensor.shape)}")
    return tensor


def reconstruct(
    model,
    input_path: Path,
    out_dir: Path,
    *,
    resolution: tuple[int, int] = (378, 672),
    pc_stride: int = 4,
    max_frames: int = 81,
    pointcloud: bool = True,
) -> dict[str, Path]:
    """Encode + decode one image/video; returns the written files by kind."""
    from scripts.demo.pointcloud_view import glb_preview, write_ply

    device = next(model.parameters()).device
    timings: dict[str, float] = {}
    started = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal started
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        now = time.perf_counter()
        timings[name] = now - started
        started = now

    frames, fps = _read_input(input_path, resolution, max_frames)
    tensor = torch.from_numpy(frames.astype(np.float32) / 255.0)
    tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0).to(device)
    lap("read")
    # bf16 like the resident I2V reference encode; decode_geometry keeps the DPT head in fp32.
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                enabled=device.type == "cuda"):
        out = model.reconstruct(tensor)
    lap("codec")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    rgb = _views_first(out["rgb"], channels=3).float().clamp(0, 1)
    rgb_frames = (rgb.permute(0, 2, 3, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    if len(rgb_frames) == 1:
        written["rgb"] = out_dir / "rgb_recon.png"
        Image.fromarray(rgb_frames[0]).save(written["rgb"])
    else:
        written["rgb"] = out_dir / "rgb_recon.mp4"
        _save_video(rgb_frames, written["rgb"], fps)
    lap("rgb_video")
    depth = out.get("depth")
    if depth is None:
        return written
    depth = _views_first(depth, channels=None).float()
    depth_cpu = depth.detach().float().cpu()
    finite = torch.isfinite(depth_cpu)
    lo = float(depth_cpu[finite].min()) if finite.any() else 0.0
    hi = float(depth_cpu[finite].max()) if finite.any() else 0.0
    depth_frames = np.stack([_depth_to_color(frame, lo, hi) for frame in depth_cpu])
    if len(depth_frames) == 1:
        written["depth"] = out_dir / "depth_recon.png"
        Image.fromarray(depth_frames[0]).save(written["depth"])
    else:
        written["depth"] = out_dir / "depth_recon.mp4"
        _save_video(depth_frames, written["depth"], fps)
    lap("depth_video")
    if pointcloud:
        if out.get("ray") is None:
            raise RuntimeError("VAE reconstruction returned no ray output; cannot export point cloud")
        from eval_generation import _scene_pointcloud_from_dpt

        views, _, h, w = rgb.shape
        xyz, colors = _scene_pointcloud_from_dpt(
            out, depth, rgb, views, h, w, rgb.device, stride=max(int(pc_stride), 1),
        )
        written["pointcloud"] = write_ply(out_dir / "recon_pointcloud.ply", xyz, colors)
        written["preview"] = glb_preview(xyz, colors, out_dir / "recon_pointcloud_preview.glb")
        print(f"pointcloud -> {written['pointcloud']} ({len(xyz)} points)", flush=True)
        lap("pointcloud")
    print(f"[vae] {len(frames)} frames: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()),
          flush=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/vae_recon"))
    parser.add_argument("--hf-repo", default="TencentARC/GAE-D64-1B")
    parser.add_argument("--cache-dir", type=Path, default=Path("ckpts"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(378, 672), metavar=("H", "W"))
    parser.add_argument("--pc-stride", type=int, default=4,
                        help="Keep every Nth pixel when exporting the reconstructed point cloud.")
    parser.add_argument("--no-pointcloud", action="store_true",
                        help="Skip reconstructed point-cloud export.")
    parser.add_argument("--all-examples", action="store_true")
    parser.add_argument("--max-frames", type=int, default=81,
                        help="Keep the first N video frames (the codec is trained on 81-frame clips).")
    args = parser.parse_args()
    if args.image is None and args.video is None and not args.all_examples:
        parser.error("provide --image, --video, or --all-examples")
    inputs = ([args.image] if args.image is not None else
              [args.video] if args.video is not None else
              sorted(Path("examples/scenes").glob("*.jpg")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAE.from_pretrained(args.hf_repo, cache_dir=args.cache_dir, device=device)
    model.eval()
    h, w = args.resolution
    for input_path in inputs:
        reconstruct(
            model, input_path, args.output / input_path.stem,
            resolution=(h, w), pc_stride=args.pc_stride,
            max_frames=args.max_frames, pointcloud=not args.no_pointcloud,
        )
        print(f"{input_path} -> {args.output / input_path.stem}")


if __name__ == "__main__":
    main()
