"""Matched resident 81-frame comparison; all ranks retain the same weights."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.demo.distributed_engine import setup
from scripts.demo.resident_engine import _atomic_json

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "ckpts")
p.add_argument("--attention", action="store_true")
p.add_argument("--scene", default="forest_lake_trail", choices=sorted(x.stem for x in (ROOT / "examples/scenes").glob("*.jpg")))
a = p.parse_args()
engine, group = setup(a.checkpoint_dir, True)
if dist.get_rank():
    group.serve_workers()
else:
    a.output.mkdir(exist_ok=False, parents=True)
    records = []
    common = dict(
        image=ROOT / f"examples/scenes/{a.scene}.jpg",
        prompt=(ROOT / f"examples/scenes/{a.scene}.txt").read_text().strip(),
        poses=ROOT / f"examples/scenes/{a.scene}_poses.npz",
        views=81,
        seed=42,
        stride=2,
    )
    for flag in (False, True):
        group.configure(
            fold_text_queries=False if a.attention else flag,
            attention_backend=("fa3" if flag else "auto") if a.attention else "auto",
        )
        engine.generate(**common, steps=2, output_dir=a.output / f"warm-{flag}")
        result = engine.generate(
            **common, steps=25, save_parity=True, output_dir=a.output / f"run-{flag}"
        )
        records.append(
            dict(
                variant=("attention" if a.attention else "text_fold"),
                candidate=flag,
                timings=result["timings"],
            )
        )
        _atomic_json(a.output / "timings.json", {"runs": records})
    before = torch.load(a.output / "run-False/parity.pt", map_location="cpu", weights_only=False)
    after = torch.load(a.output / "run-True/parity.pt", map_location="cpu", weights_only=False)
    errors = {}
    for key in ("z_ref", "plucker", "text", "null_text", "z_standardized", "rgb", "depth"):
        x, y = before[key].float(), after[key].float()
        delta = x - y
        errors[key] = dict(
            max_abs=float(delta.abs().max()),
            relative_l2=float(delta.norm() / x.norm().clamp_min(1e-8)),
            mean_abs=float(delta.abs().mean()),
        )
    rgb_error = before["rgb"].float() - after["rgb"].float()
    frame_mse = rgb_error.square().flatten(1).mean(1)
    psnr = -10 * torch.log10(frame_mse.clamp_min(1e-12))
    errors["rgb_frame_psnr_db"] = {
        "min": float(psnr.min()),
        "median": float(psnr.median()),
        "mean": float(psnr.mean()),
    }
    _atomic_json(a.output / "parity.json", errors)
    print(json.dumps(dict(runs=records, errors=errors)), flush=True)
    group.stop()
dist.destroy_process_group()
