"""Geometry ownership, z-buffer visibility and actual playable output."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.demo.progressive_preview import (
    export_ply,
    prepare,
    raster_cpu,
    render_preview,
    save_geometry,
)
from scripts.demo.render_progressive_ply import read_ply


def test_shared_splat_zbuffer_and_ties():
    # Far point's offset lands on the near point's center; it must not overwrite it.
    points = np.array([[1, 1, 1], [0, 2, 2], [1, 1, 1]], np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    image = raster_cpu(points, colors, np.eye(4, dtype=np.float32), np.eye(3), [4, 4], 4, 4, 2)
    np.testing.assert_array_equal(image[1, 1], colors[0])
    np.testing.assert_array_equal(image[1, 0], colors[1])


def test_frame_ownership_after_invalid_depth(tmp_path):
    rays = np.zeros((2, 8, 8, 6), np.float32)
    rays[..., 2] = 1
    depth = np.ones((2, 8, 8), np.float32)
    depth[1] = np.nan
    rgb = np.ones((2, 3, 8, 8), np.float32)
    path = tmp_path / "geometry.npz"
    save_geometry(
        path,
        rays,
        depth,
        rgb,
        np.repeat(np.eye(4)[None], 2, 0),
        np.repeat(np.eye(3)[None], 2, 0),
        [8, 8],
        stride=3,
    )
    with np.load(path) as d:
        assert len(d["points"]) == 9
        assert (d["frame_index"] == 0).all()
    ply = export_ply(path)
    p, c = read_ply(ply)
    assert p.shape == c.shape == (9, 3)
    assert (c == 255).all()


def test_first_view_wins_voxel_and_budget_without_voxels():
    p = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 1]], np.float32)
    c = np.array([[0, 255, 0], [255, 0, 0], [0, 0, 255]], np.uint8)
    owners = np.array([1, 0, 2])
    _, color, frame, _, _ = prepare(p, c, owners, budget=0)
    assert len(frame) == 2
    assert frame[0] == 0
    np.testing.assert_array_equal(color[0], c[1])
    assert len(prepare(p, c, owners, budget=1, voxel=0)[0]) == 1


def fixture_scene(path, views=3):
    x, y = np.meshgrid(np.linspace(-1, 1, 70), np.linspace(-0.6, 0.6, 40))
    points = np.stack([x, y, np.full_like(x, 2)], -1).reshape(-1, 3).astype(np.float32)
    colors = np.column_stack(
        [
            ((points[:, 0] + 1) * 100).astype(np.uint8),
            np.full(len(points), 120, np.uint8),
            np.full(len(points), 60, np.uint8),
        ]
    )
    cameras = np.repeat(np.eye(4, dtype=np.float32)[None], views, 0)
    cameras[:, 0, 3] = np.linspace(-0.05, 0.05, views)
    K = np.repeat(np.array([[250, 0, 160], [0, 250, 90], [0, 0, 1]], np.float32)[None], views, 0)
    np.savez(
        path,
        points=points,
        colors=colors,
        frame_index=np.arange(len(points)) % views,
        cameras=cameras,
        intrinsics=K,
        image_size=[180, 320],
    )


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_encoded_video(tmp_path):
    path = tmp_path / "fixture.npz"
    fixture_scene(path)
    video, info = render_preview(path, device="cpu", width=320, height=180)
    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)]
        )
    )
    stream = probe["streams"][0]
    assert (stream["width"], stream["height"], int(stream["nb_frames"])) == (320, 180, 3)
    assert stream["pix_fmt"] == "yuv420p"
    assert info["cumulative_points"] == sorted(info["cumulative_points"])
    assert video.with_suffix(".jpg").is_file()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("size", [1, 2, 4])
def test_scatter_raster_visibility(device, size):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from scripts.demo.progressive_preview import CudaRaster
    from scripts.demo.render_progressive_ply import NEW_POINT

    points = np.array([[1, 1, 1], [0, 2, 2], [1, 1, 1], [-1, 1, 1]], np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [100, 50, 0]], np.uint8)
    owners = np.array([0, 1, 1, 0], np.int32)
    raster = CudaRaster(points, colors, owners, device=device)
    for count in (0, len(points)):
        c = colors[:count].copy()
        c[owners[:count] == 1] = NEW_POINT
        expected = raster_cpu(points[:count], c, np.eye(4), np.eye(3), [8, 8], 8, 8, size)
        actual = raster(count, 1, np.eye(4), np.eye(3), [8, 8], 8, 8, size)
        np.testing.assert_array_equal(actual, expected)


def test_synchronized_rgb_frame_order_and_rate(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    import cv2

    source = tmp_path / "geometry.npz"
    fixture_scene(source)
    rgb = tmp_path / "rgb.mp4"
    writer = cv2.VideoWriter(str(rgb), cv2.VideoWriter_fourcc(*"mp4v"), 7, (64, 48))
    assert writer.isOpened()
    expected = [(220, 20, 20), (20, 220, 20), (20, 20, 220)]
    for color in expected:
        frame = np.empty((48, 64, 3), np.uint8)
        frame[:] = color[::-1]
        writer.write(frame)
    writer.release()
    video, timing = render_preview(
        source, tmp_path / "paired.mp4", device="cpu", width=320, height=180, rgb_video=rgb
    )
    cap = cv2.VideoCapture(str(video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(7)
    assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (640, 180)
    for color in expected:
        ok, frame = cap.read()
        assert ok
        np.testing.assert_allclose(frame[100, 160, ::-1], color, atol=12)
    assert not cap.read()[0]
    cap.release()
    assert timing["synchronized_rgb"] == str(rgb)
    with pytest.raises(ValueError, match="source frame count"):
        render_preview(source, device="cpu", rgb_video=rgb, playback_frames=81)
