"""Fast progressive previews from decoded rays, without a PLY round trip.

The compact NPZ preserves pixel/frame ownership even when depth is invalid.
Only visualization is sampled/filtered; the generation model is unchanged.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from scripts.demo.render_progressive_ply import (
    BACKGROUND,
    NEW_POINT,
    PLY_DTYPE,
    edge_mask,
    normalize_cameras,
    smooth_cameras,
    smooth_intrinsics,
)


def save_geometry(path, rays, depth, rgb, cameras, intrinsics, image_size, stride=2):
    """Persist native origin + depth * direction, with explicit frame IDs."""
    if stride < 1:
        raise ValueError("stride must be positive")
    v, h, w, channels = rays.shape
    if channels != 6 or len(depth) != v or len(rgb) != v:
        raise ValueError("inconsistent ray/depth/RGB views")
    points, colors, owners = [], [], []
    for i in range(v):
        d = np.asarray(depth[i], np.float32).reshape(depth[i].shape[-2:])
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        c = cv2.resize(np.moveaxis(rgb[i], 0, -1), (w, h), interpolation=cv2.INTER_LINEAR)
        ray = rays[i, ::stride, ::stride]
        xyz = ray[..., 3:] + d[::stride, ::stride, None] * ray[..., :3]
        valid = edge_mask(d[None], np.array([h, w]), stride, 0.05, 1).reshape(xyz.shape[:2])
        valid &= np.isfinite(xyz).all(-1) & (d[::stride, ::stride] < 100)
        points.append(xyz[valid].astype(np.float32))
        colors.append(np.clip(c[::stride, ::stride][valid] * 255, 0, 255).astype(np.uint8))
        owners.append(np.full(valid.sum(), i, np.int32))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez(
        temp,
        points=np.concatenate(points),
        colors=np.concatenate(colors),
        frame_index=np.concatenate(owners),
        cameras=np.asarray(cameras, np.float32),
        intrinsics=np.asarray(intrinsics, np.float32),
        image_size=image_size,
    )
    os.replace(temp, path)


def prepare(points, colors, owners, budget=600_000, voxel=0.001):
    valid = np.isfinite(points).all(1)
    points, colors, owners = points[valid], colors[valid], owners[valid]
    if not len(points):
        raise ValueError("No valid geometry to preview")
    center = np.median(points, axis=0)
    scale = max(float(np.percentile(np.linalg.norm(points - center, axis=1), 97)), 1e-6)
    q = (points - center) / scale
    q[:, 1] *= -1
    # Stable chronological ordering makes each voxel belong to its first view.
    order = np.argsort(owners, kind="stable")
    q, colors, owners = q[order], colors[order], owners[order]
    if voxel > 0:
        keys = np.floor((q - q.min(0)) / voxel).astype(np.int64)
        span = keys.max(0) + 1
        if float(np.prod(span.astype(np.float64))) < np.iinfo(np.int64).max:
            packed = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
            _, keep = np.unique(packed, return_index=True)
        else:
            _, keep = np.unique(keys, axis=0, return_index=True)
        keep.sort()
        q, colors, owners = q[keep], colors[keep], owners[keep]
    if budget > 0 and len(q) > budget:
        keep = np.sort(np.random.default_rng(2026).choice(len(q), budget, replace=False))
        q, colors, owners = q[keep], colors[keep], owners[keep]
    return q.astype(np.float32), colors, owners, center, scale


def raster_cpu(points, colors, camera, K, source_size, width, height, point_size=1):
    """Linear-time z-buffer, shared across all splat offsets; stable depth ties."""
    points = np.asarray(points, dtype=np.float32)
    camera = np.asarray(camera, dtype=np.float32)
    cam = (points - camera[:3, 3]) @ camera[:3, :3]
    valid = np.isfinite(cam).all(1) & (cam[:, 2] > 1e-5)
    src = np.flatnonzero(valid)
    cam = cam[valid]
    z = cam[:, 2]
    sh, sw = source_size
    x = np.rint((K[0, 0] * cam[:, 0] / z + K[0, 2]) * width / sw).astype(np.int64)
    y = np.rint((K[1, 1] * cam[:, 1] / z + K[1, 2]) * height / sh).astype(np.int64)
    pixels, depths, ids = [], [], []
    for dy in range(point_size):
        for dx in range(point_size):
            xx, yy = x + dx, y + dy
            ok = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            pixels.append(yy[ok] * width + xx[ok])
            depths.append(z[ok])
            ids.append(src[ok])
    pix, z, ids = map(np.concatenate, (pixels, depths, ids))
    zbuf = np.full(width * height, np.inf, np.float32)
    np.minimum.at(zbuf, pix, z)
    nearest = z == zbuf[pix]
    winner = np.full(width * height, len(points), np.int64)
    np.minimum.at(winner, pix[nearest], ids[nearest])
    frame = np.broadcast_to(BACKGROUND, (height * width, 3)).copy()
    occupied = winner < len(points)
    frame[occupied] = colors[winner[occupied]]
    return frame.reshape(height, width, 3)


class CudaRaster:
    """Upload once; scatter-min depth and original index keep visibility exact."""

    def __init__(self, points, colors, owners, device="cuda"):
        import torch

        self.t = torch
        self.device = device
        self.points = torch.as_tensor(points, dtype=torch.float32, device=device)
        self.colors = torch.as_tensor(colors, device=device)
        self.owners = torch.as_tensor(owners, device=device)

    def __call__(self, count, view, camera, K, source_size, width, height, point_size=1):
        t = self.t
        camera = t.as_tensor(camera, dtype=t.float32, device=self.device)
        cam = (self.points[:count] - camera[:3, 3]) @ camera[:3, :3]
        valid = t.isfinite(cam).all(1) & (cam[:, 2] > 1e-5)
        src = t.arange(count, device=self.device)[valid]
        cam = cam[valid]
        z = cam[:, 2]
        sh, sw = source_size
        x = t.round((float(K[0, 0]) * cam[:, 0] / z + float(K[0, 2])) * width / sw).long()
        y = t.round((float(K[1, 1]) * cam[:, 1] / z + float(K[1, 2])) * height / sh).long()
        pixels, depths, ids = [], [], []
        for dy in range(point_size):
            for dx in range(point_size):
                xx, yy = x + dx, y + dy
                ok = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                pixels.append(yy[ok] * width + xx[ok])
                depths.append(z[ok])
                ids.append(src[ok])
        pix, z, ids = map(t.cat, (pixels, depths, ids))
        zbuf = t.full((width * height,), float("inf"), device=self.device)
        zbuf.scatter_reduce_(0, pix, z, reduce="amin", include_self=True)
        nearest = z == zbuf[pix]
        winner = t.full((width * height,), count, dtype=t.long, device=self.device)
        winner.scatter_reduce_(0, pix[nearest], ids[nearest], reduce="amin", include_self=True)
        frame = t.as_tensor(BACKGROUND, device=self.device).expand(height * width, 3).clone()
        occupied = winner < count
        chosen = winner[occupied]
        cols = self.colors[chosen].clone()
        cols[self.owners[chosen] == view] = t.as_tensor(NEW_POINT, device=self.device)
        frame[occupied] = cols
        return frame.reshape(height, width, 3).cpu().numpy()


def overview_camera(camera, back):
    """A display camera behind and slightly above/right of the decoded camera."""
    result = camera.copy()
    right, down, forward = camera[:3, :3].T
    result[:3, 3] -= back * forward
    result[:3, 3] += .16 * right - .10 * down
    direction = camera[:3, 3] + .8 * forward - result[:3, 3]
    direction /= np.linalg.norm(direction)
    right = right - direction * np.dot(right, direction)
    right /= np.linalg.norm(right)
    down = down - direction * np.dot(down, direction) - right * np.dot(down, right)
    down /= np.linalg.norm(down)
    result[:3, :3] = np.column_stack([right, down, direction])
    return result


def fit_overview_back(cameras, intrinsics, source_size, minimum):
    """Fit every camera center with room for its frustum in the overview."""
    height, width = source_size
    centers = cameras[:, :3, 3]
    back = max(float(minimum), .08)
    for _ in range(12):
        fits = True
        for camera, K in zip(cameras, intrinsics):
            view = overview_camera(camera, back)
            points = (centers - view[:3, 3]) @ view[:3, :3]
            z = points[:, 2]
            pixels = points @ K.T
            pixels = pixels[:, :2] / np.maximum(z[:, None], 1e-5)
            if (np.any(z < .08) or np.any(pixels < [width * .10, height * .10])
                or np.any(pixels > [width * .90, height * .90])):
                fits = False
                break
        if fits:
            return back
        back *= 1.18
    return back


def overlay(
    frame, cameras, view, K, source_size, count, fresh, render_camera=None, camera_back_offset=0.08
):
    h, w = frame.shape[:2]
    sh, sw = source_size
    cam = cameras[view] if render_camera is None else render_camera

    def uv(points):
        p = (points - cam[:3, 3]) @ cam[:3, :3]
        z = p[:, 2]
        xy = np.stack(
            (
                (K[0, 0] * p[:, 0] / np.maximum(z, 1e-5) + K[0, 2]) * w / sw,
                (K[1, 1] * p[:, 1] / np.maximum(z, 1e-5) + K[1, 2]) * h / sh,
            ),
            -1,
        )
        return np.clip(xy, -10000, 10000).astype(np.int32), z > 1e-5

    # Overlay the actual decoded cameras in the same normalized scene gauge.
    centers = cameras[:, :3, 3]
    xy, valid = uv(centers)
    for i in list(range(view + 1, len(xy))) + list(range(1, view + 1)):
        if valid[i - 1] and valid[i]:
            cv2.line(frame, tuple(xy[i - 1]), tuple(xy[i]), (20, 30, 40), 5, cv2.LINE_AA)
            color = (81, 213, 255) if i <= view else (142, 169, 188)
            cv2.line(frame, tuple(xy[i - 1]), tuple(xy[i]), color, 3 if i <= view else 2, cv2.LINE_AA)
    corner = np.array([[-.022, -.014, .035], [.022, -.014, .035],
                       [.022, .014, .035], [-.022, .014, .035]], np.float32)
    for index in dict.fromkeys([0, len(cameras) - 1, view]):
        camera = cameras[index]
        origin = camera[:3, 3]
        vertices = np.vstack([origin, corner @ camera[:3, :3].T + origin])
        p, ok = uv(vertices)
        color = (255, 202, 85) if index == view else (108, 221, 164) if index == 0 else (184, 216, 255)
        for a, b in ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)):
            if ok[a] and ok[b]:
                cv2.line(frame, tuple(p[a]), tuple(p[b]), (20, 30, 40), 5, cv2.LINE_AA)
                cv2.line(frame, tuple(p[a]), tuple(p[b]), color, 2, cv2.LINE_AA)
        if ok[0]:
            cv2.circle(frame, tuple(p[0]), 4, color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (18, 16), (370, 67), (247, 250, 248), -1)
    cv2.putText(
        frame,
        f"GEOMETRY | VIEW {view + 1:02d} / {len(cameras):02d}",
        (29, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (50, 62, 59),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (29, 46), (36, 53), tuple(map(int, NEW_POINT)), -1)
    cv2.putText(
        frame,
        f"{count:,} shown | +{fresh:,} new",
        (43, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (76, 86, 81),
        1,
        cv2.LINE_AA,
    )


def render_preview(
    source,
    output=None,
    *,
    device="auto",
    width=960,
    height=540,
    budget=600_000,
    point_size=4,
    camera_back_offset=0.08,
    camera_smoothing=7,
    overview=False,
    stacked=False,
    fps=13.5,
    playback_frames=None,
    rgb_video=None,
):
    started = time.perf_counter()
    source = Path(source)
    output = Path(output) if output else source.with_suffix(".mp4")
    with np.load(source, allow_pickle=False) as data:
        points, colors, owners, center, scale = prepare(
            data["points"], data["colors"], data["frame_index"], budget
        )
        cameras = smooth_cameras(
            normalize_cameras(data["cameras"], center, scale, 0), camera_smoothing
        )
        K = smooth_intrinsics(data["intrinsics"], camera_smoothing)
        size = data["image_size"]
    if overview:
        camera_back_offset = fit_overview_back(cameras, K, size, camera_back_offset)
    capture = None
    if rgb_video is not None:
        if playback_frames not in (None, len(cameras)):
            raise ValueError("Synchronized output must preserve the source frame count")
        capture = cv2.VideoCapture(str(rgb_video))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or source_fps <= 0 or source_frames != len(cameras):
            capture.release()
            raise ValueError(
                f"RGB/geometry mismatch: {source_frames} RGB frames, {len(cameras)} views"
            )
        fps = source_fps
    output_width = width * (2 if capture is not None and not stacked else 1)
    output_height = height * (2 if capture is not None and stacked else 1)
    if width % 2 or height % 2 or point_size not in (1, 2, 4) or fps <= 0:
        raise ValueError("Use even video dimensions, positive fps, point size 1/2/4")
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    raster = CudaRaster(points, colors, owners, device) if device.startswith("cuda") else None
    prepared = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + "." + uuid.uuid4().hex + ".tmp.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{output_width}x{output_height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-threads",
        "4",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp),
    ]
    counts = np.searchsorted(owners, np.arange(len(cameras)), side="right")
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        positions = np.linspace(0, len(cameras) - 1, max(len(cameras), playback_frames or 0))
        for position in positions:
            i = int(position)
            count = counts[i]
            next_i = min(i + 1, len(cameras) - 1)
            mix = position - i
            camera = (1 - mix) * cameras[i] + mix * cameras[next_i]
            u, _, vh = np.linalg.svd(camera[:3, :3])
            camera[:3, :3] = u @ vh
            if overview:
                camera = overview_camera(camera, camera_back_offset)
            else:
                camera[:3, 3] -= camera_back_offset * camera[:3, 2]
            intrinsic = (1 - mix) * K[i] + mix * K[next_i]
            if raster is None:
                cc = colors[:count].copy()
                cc[owners[:count] == i] = NEW_POINT
                frame = raster_cpu(
                    points[:count], cc, camera, intrinsic, size, width, height, point_size
                )
            else:
                frame = raster(int(count), i, camera, intrinsic, size, width, height, point_size)
            overlay(
                frame,
                cameras,
                i,
                intrinsic,
                size,
                int(count),
                int(count - (counts[i - 1] if i else 0)),
                render_camera=camera,
                camera_back_offset=camera_back_offset,
            )
            if capture is not None:
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError(f"RGB ended before geometry view {i}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # Preserve the RGB aspect ratio with letterboxing if necessary.
                ratio = min(width / rgb.shape[1], height / rgb.shape[0])
                rw, rh = max(1, round(rgb.shape[1] * ratio)), max(1, round(rgb.shape[0] * ratio))
                panel = np.broadcast_to(BACKGROUND, (height, width, 3)).copy()
                x, y = (width - rw) // 2, (height - rh) // 2
                panel[y : y + rh, x : x + rw] = cv2.resize(
                    rgb, (rw, rh), interpolation=cv2.INTER_LINEAR
                )
                cv2.rectangle(panel, (18, 16), (210, 45), (247, 250, 248), -1)
                cv2.putText(
                    panel,
                    f"RGB | VIEW {i + 1:02d} / {len(cameras):02d}",
                    (29, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (50, 62, 59),
                    1,
                    cv2.LINE_AA,
                )
                frame = np.concatenate([panel, frame], axis=0 if stacked else 1)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        error = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"Preview encoding failed: {error[-2000:]}")
        os.replace(temp, output)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stderr.close()
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        temp.unlink(missing_ok=True)
        if capture is not None:
            capture.release()
    cv2.imwrite(str(output.with_suffix(".jpg")), frame[..., ::-1])
    timing = {
        "device": device,
        "width": output_width,
        "synchronized_rgb": str(rgb_video) if rgb_video is not None else None,
        "height": output_height,
        "views": len(cameras),
        "video_frames": len(positions),
        "rendered_points": len(points),
        "point_size": point_size,
        "camera_back_offset": camera_back_offset,
        "camera_smoothing": camera_smoothing,
        "overview": overview,
        "point_budget": budget,
        "cumulative_points": counts.tolist(),
        "prepare_seconds": prepared - started,
        "render_encode_seconds": time.perf_counter() - prepared,
        "total_seconds": time.perf_counter() - started,
        "fps": fps,
    }
    output.with_suffix(".json").write_text(json.dumps(timing, indent=2) + "\n")
    return output, timing


def export_ply(source):
    """Optional binary download, generated only when requested by the user."""
    source = Path(source)
    output = source.with_suffix(".ply")
    with np.load(source, allow_pickle=False) as data:
        p, c = data["points"], data["colors"]
    vertices = np.empty(len(p), PLY_DTYPE)
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = p[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        vertices[name] = c[:, i]
    temp = output.with_name(output.stem + "." + uuid.uuid4().hex + ".tmp.ply")
    with temp.open("wb") as f:
        f.write(
            (
                "ply\nformat binary_little_endian 1.0\nelement vertex " + str(len(p)) + "\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            ).encode()
        )
        f.write(vertices.tobytes())
    os.replace(temp, output)
    return output
