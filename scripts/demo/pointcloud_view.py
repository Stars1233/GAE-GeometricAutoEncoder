"""Point-cloud files for the browser: fast binary PLY and a Model3D-ready GLB."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> Path:
    """Binary little-endian XYZRGB PLY (the ASCII writer takes seconds per million points)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, axis in enumerate("xyz"):
        rows[axis] = xyz[:, i]
    for i, channel in enumerate(("red", "green", "blue")):
        rows[channel] = rgb[:, i]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(rows)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        rows.tofile(handle)
    return path


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """XYZ float32 and RGB uint8 from an XYZRGB PLY (ASCII or binary little-endian)."""
    with open(path, "rb") as handle:
        count, binary = 0, False
        while True:
            line = handle.readline()
            if not line:
                return None
            if line.startswith(b"format"):
                binary = b"binary_little_endian" in line
            if line.startswith(b"element vertex"):
                count = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        if count == 0:
            return None
        if binary:
            rows = np.fromfile(handle, count=count, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
            return rows["xyz"], rows["rgb"]
        data = np.fromfile(handle, sep=" ", dtype=np.float32)
    if data.size < count * 6:
        return None
    data = data[: count * 6].reshape(count, 6)
    return data[:, :3], data[:, 3:6].astype(np.uint8)


def glb_preview(xyz: np.ndarray, rgb: np.ndarray, out: Path, max_points: int = 200_000) -> Path:
    """Model3D renders .ply as a Gaussian splat, so show an RGB point GLB instead."""
    import trimesh

    keep = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > max_points:
        pick = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[pick], rgb[pick]
    # The viewer frames the whole bounding box, so a few far points (sky)
    # would shrink the scene to a dot; keep the 1-99% core with a margin.
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    margin = 0.1 * (hi - lo)
    core = np.all((xyz >= lo - margin) & (xyz <= hi + margin), axis=1)
    if core.sum() >= 1000:
        xyz, rgb = xyz[core], rgb[core]
    xyz = xyz - np.median(xyz, axis=0)
    xyz = xyz / max(float(np.abs(xyz).max()), 1e-6)
    xyz[:, 1:] *= -1  # OpenCV camera frame (y down, z forward) -> glTF (y up)
    colors = np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], axis=1)
    out = Path(out)
    trimesh.PointCloud(xyz.astype(np.float32), colors=colors).export(out, file_type="glb")
    return out


def ply_preview(ply: Path, max_points: int = 200_000) -> Path | None:
    cloud = read_ply(ply)
    if cloud is None:
        return None
    return glb_preview(*cloud, Path(ply).with_name(Path(ply).stem + "_preview.glb"), max_points)
