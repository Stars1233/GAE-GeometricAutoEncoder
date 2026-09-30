"""Resident inference: rank 0 owns conditioning/decoders, all ranks sample.

Launch with torchrun --standalone --nproc_per_node=8. No external task submission.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "scripts" / "eval"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.distributed as dist

from scripts.demo.resident_engine import Engine, _atomic_json
from stage2.models import ulysses


class SamplerGroup:
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.busy = False
        # Idle workers wait on CPU; an idle NCCL broadcast would trigger its
        # watchdog and take down an otherwise healthy resident service.
        self.control = dist.new_group(backend="gloo", timeout=timedelta(hours=48))

    def _broadcast(self, values=None):
        meta = None
        if dist.get_rank() == 0:
            tensors = {k: v for k, v in values.items() if torch.is_tensor(v)}
            meta = dict(
                options={k: v for k, v in values.items() if k not in tensors},
                tensors={
                    k: [list(v.shape), str(v.dtype).split(".")[-1]] for k, v in tensors.items()
                },
            )
        messages = [meta]
        dist.broadcast_object_list(messages, src=0, group=self.control)
        meta = messages[0]
        if meta["options"].get("stop"):
            return meta["options"]
        result = dict(meta["options"])
        for key, (shape, dtype) in meta["tensors"].items():
            tensor = (
                values[key].contiguous()
                if dist.get_rank() == 0
                else torch.empty(shape, dtype=getattr(torch, dtype), device=self.device)
            )
            dist.broadcast(tensor, src=0)
            result[key] = tensor
        return result

    def __call__(self, model, z_ref, views, cond_num, **kwargs):
        from eval_generation import sample_v4_euler

        values = dict(kwargs, z_ref_clean=z_ref, total_view=views, cond_num=cond_num)
        self.busy = True
        self._broadcast(values)
        result = sample_v4_euler(model, **values)
        self.busy = False
        return result

    def configure(self, *, fold_text_queries=None, attention_backend=None):
        if self.busy:
            raise RuntimeError("Cannot change inference options during sampling")
        options = {}
        if fold_text_queries is not None:
            options["fold_text_queries"] = bool(fold_text_queries)
        if attention_backend is not None:
            options["attention_backend"] = attention_backend
        self._validate_configuration(options)
        self._broadcast({"configure": options})
        self._apply_configuration(options)

    @staticmethod
    def _validate_configuration(options):
        if set(options) - {"fold_text_queries", "attention_backend"}:
            raise ValueError("Unsupported inference configuration")
        if "fold_text_queries" in options and type(options["fold_text_queries"]) is not bool:
            raise ValueError("fold_text_queries must be boolean")
        if options.get("attention_backend", "auto") not in {"auto", "fa3"}:
            raise ValueError("Unsupported attention backend")

    def _apply_configuration(self, options):
        self._validate_configuration(options)
        for block in self.model.blocks:
            if "fold_text_queries" in options:
                block.fold_text_queries = options["fold_text_queries"]
            if "attention_backend" in options:
                block.attn.attention_backend = options["attention_backend"]

    def serve_workers(self):
        from eval_generation import sample_v4_euler

        while True:
            values = self._broadcast()
            if values.get("stop"):
                return
            if "configure" in values:
                self._apply_configuration(values["configure"])
                continue
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sample_v4_euler(self.model, **values)

    def stop(self):
        if self.busy:
            raise RuntimeError(
                "Cannot stop through metadata while a sampler collective is incomplete"
            )
        self._broadcast(dict(stop=True))


def setup(checkpoint_dir, cfg_parallel):
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    ulysses.init_ulysses(cfg_parallel=cfg_parallel)
    device = torch.device("cuda", local)
    if dist.get_rank() == 0:
        engine = Engine(checkpoint_dir, device)
        model = engine.model.flow
    else:
        from gae.pipeline import load_flow

        model = load_flow(
            str(ROOT / "configs/flow_gae64.yaml"),
            str(Path(checkpoint_dir) / "flow_gae64.pt"),
            device,
        )
        model.fast_inference = True
        engine = None
    group = SamplerGroup(model, device)
    if engine is not None:
        engine.sampler = group
    dist.barrier(device_ids=[local])
    return engine, group


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serve", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--cfg-parallel", action="store_true")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--poses")
    p.add_argument("--views", type=int, default=81)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-parity", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    engine, group = setup(a.checkpoint_dir, a.cfg_parallel)
    if dist.get_rank() != 0:
        group.serve_workers()
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        report = {
            "status": "running",
            "world_size": dist.get_world_size(),
            "cfg_parallel": a.cfg_parallel,
            "requests": [],
        }
        _atomic_json(a.output / "benchmark.json", report)
        common = dict(
            image=a.image,
            prompt=Path(a.prompt_file).read_text().strip(),
            poses=a.poses,
            views=a.views,
            seed=a.seed,
            stride=2,
        )
        try:
            if a.warmup_steps:
                report["warmup"] = engine.generate(
                    **common, steps=a.warmup_steps, output_dir=a.output / "warmup"
                )
                _atomic_json(a.output / "benchmark.json", report)
            if a.serve:
                import camera_studio as app

                app.RESIDENT_ENGINE = engine
                report["status"] = "ready"
                _atomic_json(a.output / "benchmark.json", report)
                app.demo.queue(default_concurrency_limit=1).launch(
                    server_name=a.host,
                    server_port=a.port,
                    share=False, css=app.CSS,
                    allowed_paths=[str(app.OUTPUT_ROOT), str(ROOT / "examples"),
                                   str(ROOT / "scripts/demo/camera_editor.js"),
                                   str(ROOT / "scripts/demo/camera_studio.css")],
                )
            for i in range(0 if a.serve else a.repeat):
                result = engine.generate(
                    **common,
                    steps=a.steps,
                    output_dir=a.output / f"request-{i:03d}",
                    save_parity=a.save_parity,
                )
                report["requests"].append(result)
                _atomic_json(a.output / "benchmark.json", report)
                print(json.dumps(result), flush=True)
            report["status"] = "complete"
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _atomic_json(a.output / "benchmark.json", report)
            if not group.busy:
                group.stop()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
