"""Resident, single-image GAE inference using the released evaluator semantics.

No weights are downloaded. Launch from a provisioned checkout, for example::

    python scripts/demo/resident_engine.py --checkpoint-dir /path/to/ckpts \
      --image examples/scenes/forest_lake_trail.jpg \
      --prompt-file examples/scenes/forest_lake_trail.txt \
      --poses examples/scenes/forest_lake_trail_poses.npz \
      --output results/resident-benchmark --steps 25 --repeat 2

Initialization and optional warmup happen once, before measured requests.
The returned request time includes synchronized RGB/geometry video encoding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src", ROOT / "scripts" / "eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reference_path(image: Path) -> Path | None:
    candidates = (
        image.with_name(image.stem + "_poses.npz"),
        ROOT / "examples" / "scenes" / (image.stem + "_poses.npz"),
        ROOT / "examples" / "scenes" / "forest_lake_trail_poses.npz",
    )
    return next((p for p in candidates if p.is_file()), None)


def _prepare_input(image, poses, trajectory, views, seed, reference_poses=None):
    """Bypass lossless video IO, retaining wrapper resize and evaluator crop/K."""
    import cv2
    import numpy as np
    from eval_data import _IMG_NORM, _crop_resize_if_necessary

    from scripts.demo.generate import _load_gt_cameras
    from scripts.demo.trajectory_utils import (
        load_reference_poses,
        reference_forward_sign,
        synthesize_free_trajectory,
        trajectory_extent,
    )

    image = Path(image).resolve()
    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read input image: {image}")
    rgb = cv2.cvtColor(cv2.resize(bgr, (672, 378), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    reference = Path(reference_poses) if reference_poses else _reference_path(image)
    if poses is not None or trajectory == "example":
        chosen = Path(poses) if poses is not None else reference
        if chosen is None:
            raise ValueError("Example trajectory requires camera poses")
        c2w, intrinsics, fps = _load_gt_cameras(chosen, views)
        if len(c2w) != views:
            raise ValueError(f"Expected {views} camera poses, found {len(c2w)} in {chosen}")
        mode = f"poses:{chosen.resolve()}"
    else:
        if trajectory not in ("forward", "backward", "turn_left", "turn_right"):
            raise ValueError(f"Unsupported trajectory: {trajectory}")
        focal = 0.8 * 672
        K = np.array([[focal, 0.0, 336.0], [0.0, focal, 189.0], [0.0, 0.0, 1.0]], np.float32)
        intrinsics = np.repeat(K[None], views, axis=0)
        kwargs = {}
        if reference is not None:
            ref = load_reference_poses(reference, views)
            extent = trajectory_extent(ref)
            if extent > 0:
                # Match generate.py's serialized 9-digit rollout settings.
                kwargs = dict(
                    target_extent=float(f"{extent:.9g}"),
                    target_extents_xyz=np.array(
                        [float(f"{v:.9g}") for v in np.ptp(ref[:, :3, 3], axis=0)]
                    ),
                    target_direction=np.array(
                        [float(f"{v:.9g}") for v in ref[-1, :3, 3] - ref[0, :3, 3]]
                    ),
                    fwd_sign=reference_forward_sign(ref),
                )
        c2w = np.asarray(
            synthesize_free_trajectory(
                np.eye(4), views, motion=trajectory, speed=0.06, seed=seed, **kwargs
            )
        )
        fps, mode = 12, f"synthetic:{trajectory}"
    # Every view's principal point can change the crop. Only reference pixels
    # enter the encoder, but each camera needs its own adjusted intrinsics.
    # Editor cameras already refer to the prepared pixels; cropping again
    # would change the reference image and its principal point.
    if poses is not None:
        with np.load(poses, allow_pickle=False) as data:
            if bool(data.get("image_preprocessed", False)):
                return _IMG_NORM(rgb).unsqueeze(0), intrinsics, c2w, int(fps or 12), mode
    adjusted = []
    image_tensor = None
    for i, K in enumerate(intrinsics):
        cropped, new_K = _crop_resize_if_necessary(rgb, K.copy(), (672, 378))
        adjusted.append(new_K)
        if i == 0:
            image_tensor = _IMG_NORM(cropped).unsqueeze(0)
    return image_tensor, np.asarray(adjusted), np.asarray(c2w), int(fps or 12), mode


def _is_editor_camera(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        return bool(data.get("image_preprocessed", False))


class Engine:
    """One resident model bundle; requests are serialized, never silently cached."""

    def __init__(self, checkpoint_dir, device="cuda:0", *, qwen_model=None):
        started = time.perf_counter()
        import eval_generation as ev
        import torch
        from omegaconf import OmegaConf

        from gae import GAE
        from gae.pipeline import _CODEC_STATE_KEYS, _FLOW_STATE_KEYS, _unwrap_state
        from stage2.models.text_encoder import Qwen3TextEncoder

        self.torch, self.ev = torch, ev
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                "Resident inference requires a CUDA device; CPU tests use helpers only"
            )
        torch.cuda.set_device(self.device)
        self.lock = threading.Lock()
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        paths = {
            name: self.checkpoint_dir / name
            for name in ("gae_64.pt", "flow_gae64.pt", "latent_stats_gae_64.pt")
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        # The service is provisioned separately. Missing Hub caches must fail
        # at startup rather than performing an unbounded download on a click.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        flow_cfg = OmegaConf.load(ROOT / "configs/flow_gae64.yaml")
        codec_cfg = OmegaConf.load(ROOT / "configs/gae_64.yaml")
        for cfg in (flow_cfg, codec_cfg):
            if cfg.get("codec", {}).get("stats_dir"):
                cfg.codec.stats_dir = str((ROOT / cfg.codec.stats_dir).resolve())
            params = cfg.get("stage_1", {}).get("params", {})
            for key in ("da3_weights_path", "dpt_decoder_path"):
                value = params.get(key)
                if value and value != "none" and not Path(value).is_absolute():
                    params[key] = str(ROOT / value)
        flow_cfg.stage_1.params.encoder_pretrained_path = ev._resolve_da3_encoder_path(
            flow_cfg.stage_1.params.encoder_pretrained_path,
            flow_cfg.stage_1.params.da3_weights_path,
        )
        with tempfile.TemporaryDirectory(prefix="gae-resident-config-") as folder:
            flow_path, codec_path = Path(folder) / "flow.yaml", Path(folder) / "codec.yaml"
            OmegaConf.save(flow_cfg, flow_path)
            OmegaConf.save(codec_cfg, codec_path)
            self.model = GAE.from_configs(
                str(codec_path),
                str(paths["gae_64.pt"]),
                flow_cfg=str(flow_path),
                flow_ckpt=str(paths["flow_gae64.pt"]),
                latent_stats=str(paths["latent_stats_gae_64.pt"]),
                device=self.device,
            ).eval()
        self.model.requires_grad_(False)
        self.model.flow.fast_inference = True
        # Public loaders allow unexpected keys. A serving engine must reject a
        # mismatched checkpoint rather than advertising a partially loaded model.
        for name, module, wrappers in (
            ("gae_64.pt", self.model.codec, _CODEC_STATE_KEYS),
            ("flow_gae64.pt", self.model.flow, _FLOW_STATE_KEYS),
        ):
            state = _unwrap_state(
                torch.load(paths[name], map_location="cpu", weights_only=False), wrappers
            )
            expected, found = set(module.state_dict()), set(state)
            missing = sorted(k for k in expected - found if not k.startswith("repa_proj"))
            unexpected = sorted(k for k in found - expected if not k.startswith("repa_proj."))
            if missing or unexpected:
                raise ValueError(
                    f"{name}: missing keys {missing[:12]}, unexpected keys {unexpected[:12]}"
                )
            del state
        stats = torch.load(paths["latent_stats_gae_64.pt"], map_location="cpu", weights_only=False)
        self.mean = stats["mean"].float().to(self.device).reshape(1, -1, 1, 1)
        self.std = stats["std"].float().to(self.device).reshape(1, -1, 1, 1).clamp(min=1e-5)
        if ("whiten" in stats) != ("unwhiten" in stats):
            raise ValueError("Latent statistics need both whiten and unwhiten")
        self.whiten = stats["whiten"].float().to(self.device) if "whiten" in stats else None
        self.unwhiten = stats["unwhiten"].float().to(self.device) if "unwhiten" in stats else None
        if self.model.codec.rgb_head is None or self.model.backbone.rae_cl_decoder is None:
            raise ValueError("Released RGB and geometry decoders are both required")
        self.config = flow_cfg
        text_model = qwen_model or ev._resolve_text_model_path(
            str(flow_cfg.text_encoder.model_name)
        )
        self.text_encoder = (
            Qwen3TextEncoder(
                model_name=str(text_model),
                max_length=int(flow_cfg.text_encoder.max_length),
                torch_dtype=torch.bfloat16,
            )
            .to(self.device)
            .eval()
        )
        self.text_encoder.requires_grad_(False)
        with torch.inference_mode():
            self.null_tokens = self.text_encoder([""])["tokens"].to(self.device)
        self._sync()
        self.initialization_seconds = time.perf_counter() - started

    def prepare_camera_scene(self, image):
        """Single-image proxy calibrated to metric camera translation units."""
        import numpy as np
        import cv2
        from scripts.demo.direct_camera import image_key, metric_depth_scale
        t, ev, rae = self.torch, self.ev, self.model.backbone
        with self.lock, t.inference_mode():
            images, _, _, _, _ = _prepare_input(image, None, "forward", 1, 42)
            with t.autocast("cuda", dtype=t.bfloat16):
                features = rae.encode(images.to(self.device).unsqueeze(0), mode="all")
            features = {k: v.float() for k, v in features.items()}
            decoded = ev.decode_dpt(ev.raw_to_dpt_input(
                features, rae.encoder.backbone.pretrained.norm, 1), rae.rae_cl_decoder, 378, 672)
            depth = decoded["depth"].float().cpu().numpy().reshape(378, 672)
            # DA3 depth parameterizes its native rays; recover source-camera Z
            # before backprojecting through the estimated source intrinsics.
            ray = ev._ray_to_numpy(decoded["ray"])
            cameras, native_K = ev.recover_poses(ray, ev._rayconf_to_numpy(decoded.get("ray_conf")),
                                        input_size=(378, 672), return_per_view_intrinsics=True)
            ray_depth = cv2.resize(depth, (ray.shape[2], ray.shape[1]))
            points = ray[0, ..., 3:] + ray_depth[..., None] * ray[0, ..., :3]
            camera = np.asarray(cameras[0])
            source_points = (points - camera[:3, 3]) @ camera[:3, :3]
            depth = cv2.resize(source_points[..., 2], (672, 378))
            # Flow was trained with metric translations. The any-view encoder
            # has arbitrary scene scale; calibrate it with the cached metric head.
            K = np.asarray(native_K[0], dtype=np.float32)
            if not hasattr(self, "camera_metric_model"):
                from depth_anything_3.api import DepthAnything3
                self.camera_metric_model = DepthAnything3.from_pretrained(
                    "depth-anything/DA3METRIC-LARGE").to(self.device).eval().requires_grad_(False)
            metric = self.camera_metric_model(images.to(self.device).unsqueeze(0), export_feat_layers=[])
            metric_prediction = metric["depth"].float().cpu().numpy().reshape(378, 672)
            sky = metric.get("sky")
            sky = sky.float().cpu().numpy().reshape(378, 672) if sky is not None else None
            metric_scale, scale_quantiles = metric_depth_scale(depth, metric_prediction, K, sky)
            depth *= metric_scale
            # Keep photo colors; the geometry decoder supplies only depth.
            rgb = images[0].permute(1, 2, 0).numpy()
            rgb = np.clip(rgb * np.array([.229, .224, .225]) + np.array([.485, .456, .406]), 0, 1)
            prepared_dir = ROOT / ".gradio_previews"
            prepared_dir.mkdir(exist_ok=True)
            prepared_image = prepared_dir / f"source-{image_key(image)}.png"
            if not cv2.imwrite(str(prepared_image), cv2.cvtColor(np.rint(rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)):
                raise OSError("Could not save the prepared camera image.")
            yy, xx = np.mgrid[0:378:3, 0:672:3]
            z = depth[::3, ::3]
            xyz = np.stack(((xx-K[0,2])*z/K[0,0], (yy-K[1,2])*z/K[1,1], z), -1)
            valid = np.isfinite(xyz).all(-1) & (z > 0)
            if not valid.any():
                raise ValueError("Could not estimate depth for this image.")
            center = depth[126:252, 224:448]
            center = center[np.isfinite(center) & (center > 0)]
            pivot = float(np.median(center if center.size else z[valid]))
            return dict(positions=np.round(xyz[valid], 5).tolist(),
                        colors=np.round(rgb[::3, ::3][valid], 4).tolist(),
                        width=672, height=378, K=K.tolist(), pivotDepth=pivot,
                        imageKey=image_key(image), preparedImage=str(prepared_image),
                        metricScale=metric_scale, scaleQuantiles=scale_quantiles, depthUnits="estimated meters")

    def text_to_image(self, prompt, *, steps=25, guidance="ig", scale=2.0, seed=0, pc_stride=4,
                      output_dir):
        """One 672x378 image with depth and point cloud, sampled like ``generate_t2i.py``.

        ``guidance`` is ``ig`` (internal guidance, the script default) or ``cfg``;
        ``scale`` is the matching guidance strength.
        """
        if guidance not in ("ig", "cfg"):
            raise ValueError("guidance must be 'ig' or 'cfg'")
        from eval_data import depth_to_numpy_img
        from PIL import Image

        from scripts.demo.pointcloud_view import glb_preview, write_ply

        if not prompt or not prompt.strip():
            raise ValueError("A nonempty prompt is required")
        t, ev = self.torch, self.ev
        vae, rae, dit = self.model.codec, self.model.backbone, self.model.flow
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        with self.lock, t.inference_mode():
            text = self.text_encoder([prompt.strip()])["tokens"].to(self.device)
            t.manual_seed(int(seed))
            with t.autocast("cuda", dtype=t.bfloat16):
                z_std = getattr(self, "sampler", ev.sample_v4_euler)(
                    dit,
                    None,
                    1,
                    0,
                    plucker_6d=None,
                    ref_global=text,
                    cfg_uncond_ref_global=self.null_tokens,
                    num_steps=int(steps),
                    cfg_scale=float(scale) if guidance == "cfg" else 1.0,
                    guidance_mode=guidance,
                    ig_scale=float(scale) if guidance == "ig" else 0.0,
                    time_dist_shift=float(self.config.misc.time_dist_shift),
                    eps=0.001,
                    noise_shape=(int(self.mean.shape[1]), 378 // 14, 672 // 14),
                    device=self.device,
                    dtype=t.float32,
                    prediction=str(self.config.transport.params.prediction),
                )
                z = ev._denorm_latent(z_std, self.mean, self.std, self.unwhiten)
                rgb = self.model.decode_rgb(z, num_views=1).float().clamp(0, 1)
                seq, h, w = vae._decode_trunk(z)
                raw = vae.dec_conv(seq.permute(0, 2, 1).reshape(-1, seq.shape[-1], h, w))
                features = vae.denormalize_and_split(raw)
            dpt = ev.decode_dpt(
                ev.raw_no_cls_to_dpt_input(features, rae.encoder.backbone.pretrained.norm, 1),
                rae.rae_cl_decoder,
                378,
                672,
            )
            image = output / "t2i.png"
            Image.fromarray(
                (rgb[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype("uint8")
            ).save(image)
            depth = output / "t2i_depth.png"
            Image.fromarray(depth_to_numpy_img(dpt["depth"][0])).save(depth)
            xyz, colors = ev._scene_pointcloud_from_dpt(
                dpt, dpt["depth"], rgb, 1, 378, 672, self.device, stride=max(int(pc_stride), 1)
            )
        pointcloud = write_ply(output / "t2i_pointcloud.ply", xyz, colors)
        preview = glb_preview(xyz, colors, output / "t2i_pointcloud_preview.glb")
        return dict(image=image, depth=depth, pointcloud=pointcloud, preview=preview)

    def _sync(self):
        self.torch.cuda.synchronize(self.device)

    @contextmanager
    def _phase(self, timings, name):
        self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            timings[name] = time.perf_counter() - started

    def generate(
        self,
        image,
        prompt,
        *,
        poses=None,
        trajectory="example",
        views=81,
        steps=25,
        cfg_scale=2.0,
        seed=42,
        stride=2,
        output_dir,
        reference_poses=None,
        save_parity=False,
        render_device=None,
    ):
        """Generate fresh RGB and synchronized cloud video in a new directory.

        ``save_parity`` saves model inputs/latents and uncompressed RGB/depth
        tensors for numerical comparison; its IO is included in request time.
        """
        requested = time.perf_counter()
        if not prompt or not prompt.strip():
            raise ValueError("A nonempty prompt is required")
        if views < 2 or steps < 1 or stride < 1:
            raise ValueError("views >= 2, steps >= 1 and stride >= 1 are required")
        with self.lock:
            output = Path(output_dir).resolve()
            output.mkdir(parents=True, exist_ok=False)
            timings = {"queue_seconds": time.perf_counter() - requested}
            try:
                result = self._generate(
                    image,
                    prompt.strip(),
                    poses,
                    trajectory,
                    int(views),
                    int(steps),
                    float(cfg_scale),
                    int(seed),
                    int(stride),
                    output,
                    reference_poses,
                    save_parity,
                    render_device,
                    timings,
                )
                self._sync()
                result["timings"]["request_seconds"] = time.perf_counter() - requested
                _atomic_json(output / "result.json", result)
                return result
            except BaseException as exc:
                _atomic_json(
                    output / "failure.json",
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "timings": timings,
                        "request_seconds": time.perf_counter() - requested,
                    },
                )
                raise

    def _generate(
        self,
        image,
        prompt,
        poses,
        trajectory,
        views,
        steps,
        cfg_scale,
        seed,
        stride,
        output,
        reference_poses,
        save_parity,
        render_device,
        timings,
    ):
        import numpy as np
        from eval_data import tensor_to_numpy_img

        from scripts.demo.progressive_preview import render_preview

        t, ev = self.torch, self.ev
        vae, rae, dit = self.model.codec, self.model.backbone, self.model.flow
        training = self.config.training
        with self._phase(timings, "input_seconds"):
            images, Ks, c2w, fps, mode = _prepare_input(
                image, poses, trajectory, views, seed, reference_poses
            )
            images = images.to(self.device)
        with t.inference_mode():
            t.manual_seed(seed)
            with self._phase(timings, "condition_seconds"), t.autocast("cuda", dtype=t.bfloat16):
                z_ref = ev._encode_ref_latent(
                    rae, vae, images, 378, 672, self.device, self.mean, self.std, self.whiten
                )
                patch = int(getattr(dit, "s_patch_size", 1))
                plucker = ev.build_plucker_from_cameras(
                    Ks,
                    c2w,
                    378,
                    672,
                    z_ref.shape[-2] // patch,
                    z_ref.shape[-1] // patch,
                    self.device,
                    origin_idx=0 if training.pose_origin == "first" else -1,
                    translation_norm=str(training.pose_translation_norm),
                    pose_scale=None,
                )
            with self._phase(timings, "text_seconds"):
                text = self.text_encoder([prompt])["tokens"].to(self.device)
            with self._phase(timings, "sample_seconds"), t.autocast("cuda", dtype=t.bfloat16):
                guidance = self.config.validation.guidance
                z_std = getattr(self, "sampler", ev.sample_v4_euler)(
                    dit,
                    z_ref,
                    views,
                    1,
                    plucker_6d=plucker,
                    ref_global=text,
                    cfg_uncond_ref_global=self.null_tokens,
                    num_steps=steps,
                    cfg_scale=cfg_scale,
                    cfg_interval=(float(guidance.t_min), float(guidance.t_max)),
                    guidance_mode="cfg",
                    ig_scale=2.0,
                    time_dist_shift=float(self.config.misc.time_dist_shift),
                    eps=0.001,
                    prediction=str(self.config.transport.params.prediction),
                    reference_conditioning=str(
                        training.get("reference_conditioning", "clean_token_v4")
                    ),
                    baseline_cam_kwargs=None,
                    noise_corr=0.0,
                    view_frame_idx=t.arange(views, device=self.device),
                )
                z = ev._denorm_latent(z_std, self.mean, self.std, self.unwhiten)
            with self._phase(timings, "trunk_seconds"), t.autocast("cuda", dtype=t.bfloat16):
                seq, h, w = vae._decode_trunk(z)
            with self._phase(timings, "rgb_seconds"), t.autocast("cuda", dtype=t.bfloat16):
                rgb_raw = vae.rgb_head(seq, h, w, num_views=views)
                rgb = (rgb_raw * rae.encoder_std.squeeze(0) + rae.encoder_mean.squeeze(0)).clamp(
                    0, 1
                )
            with self._phase(timings, "geometry_seconds"):
                with t.autocast("cuda", dtype=t.bfloat16):
                    raw = vae.dec_conv(seq.permute(0, 2, 1).reshape(-1, seq.shape[-1], h, w))
                    features = vae.denormalize_and_split(raw)
                dpt = ev.decode_dpt(
                    ev.raw_no_cls_to_dpt_input(
                        features, rae.encoder.backbone.pretrained.norm, views
                    ),
                    rae.rae_cl_decoder,
                    378,
                    672,
                )
            source, video = output / "000_preview.npz", output / "000_pred.mp4"
            with self._phase(timings, "geometry_export_seconds"):
                ev._save_progressive_geometry(dpt, dpt["depth"], rgb, str(source), 378, 672, stride)
            with self._phase(timings, "rgb_encode_seconds"):
                ev._save_mp4([tensor_to_numpy_img(frame) for frame in rgb], str(video), fps=fps)
                if not video.is_file() or not video.stat().st_size:
                    raise RuntimeError("RGB encoder did not produce a video")
            depth_video = output / "000_depth.mp4"
            with self._phase(timings, "depth_encode_seconds"):
                ev._save_mp4(ev.depth_to_numpy_video(dpt["depth"]), str(depth_video), fps=fps)
            if save_parity:
                with self._phase(timings, "parity_export_seconds"):
                    t.save(
                        {
                            "z_ref": z_ref.cpu(),
                            "z_standardized": z_std.cpu(),
                            "z": z.cpu(),
                            "image_normalized": images.cpu(),
                            "plucker": plucker.cpu(),
                            "text": text.cpu(),
                            "null_text": self.null_tokens.cpu(),
                            "rgb": rgb.cpu(),
                            "depth": dpt["depth"].cpu(),
                            "K": Ks,
                            "c2w": c2w,
                            "seed": seed,
                        },
                        output / "parity.pt",
                    )
            # Drop request activations before the GPU renderer allocates its buffers.
            del dpt, features, raw, seq, rgb_raw, rgb, z, z_std, z_ref, text, plucker
            editor_camera = poses is not None and _is_editor_camera(poses)
            with self._phase(timings, "preview_seconds"):
                synchronized, preview_timing = render_preview(
                    source,
                    output / "000_synchronized.mp4",
                    rgb_video=video,
                    device=str(render_device or self.device),
                    fps=fps,
                    camera_back_offset=.35, overview=True, stacked=True,
                    camera_smoothing=1 if editor_camera else 7,
                )
        return {
            "rgb_video": str(video),
            "depth_video": str(depth_video) if depth_video.is_file() else None,
            "synchronized_video": str(synchronized),
            "geometry": str(source),
            "views": views,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "stride": stride,
            "mode": mode,
            "resolution": [672, 378],
            "fps": fps,
            "device": str(self.device),
            "world_size": t.distributed.get_world_size() if t.distributed.is_initialized() else 1,
            "attention_backend": getattr(dit.blocks[0].attn, "attention_backend", "auto"),
            "initialization_seconds": self.initialization_seconds,
            "timings": timings,
            "preview": preview_timing,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--qwen-model")
    p.add_argument("--image", type=Path, required=True)
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    p.add_argument("--poses", type=Path)
    p.add_argument(
        "--trajectory",
        default="example",
        choices=("example", "forward", "backward", "turn_left", "turn_right"),
    )
    p.add_argument("--reference-poses", type=Path)
    p.add_argument("--views", type=int, default=81)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--save-parity", action="store_true")
    p.add_argument("--render-device")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.repeat < 1 or a.warmup_steps < 0:
        p.error("--repeat must be positive and --warmup-steps nonnegative")
    a.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "initializing", "requests": []}
    report_path = a.output / "benchmark.json"
    _atomic_json(report_path, report)
    try:
        engine = Engine(a.checkpoint_dir, a.device, qwen_model=a.qwen_model)
        common = dict(
            image=a.image,
            prompt=a.prompt or a.prompt_file.read_text().strip(),
            poses=a.poses,
            trajectory=a.trajectory,
            reference_poses=a.reference_poses,
            views=a.views,
            cfg_scale=a.cfg_scale,
            seed=a.seed,
            stride=a.stride,
            render_device=a.render_device,
        )
        report["initialization_seconds"] = engine.initialization_seconds
        if a.warmup_steps:
            report["warmup"] = engine.generate(
                **common, steps=a.warmup_steps, output_dir=a.output / "warmup"
            )
        report["status"] = "ready"
        _atomic_json(report_path, report)
        print("[resident] ready", flush=True)
        for i in range(a.repeat):
            result = engine.generate(
                **common,
                steps=a.steps,
                save_parity=a.save_parity,
                output_dir=a.output / f"request-{i:03d}",
            )
            report["requests"].append(result)
            _atomic_json(report_path, report)
            print(json.dumps(result), flush=True)
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _atomic_json(report_path, report)


if __name__ == "__main__":
    main()
