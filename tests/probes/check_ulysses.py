"""Run with torchrun --standalone --nproc_per_node=8; no model downloads."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from stage2.models import ulysses as u
from stage2.models.dit_temporal import GAEFlowTemporal

p = argparse.ArgumentParser()
p.add_argument("--cfg-parallel", action="store_true")
p.add_argument("--output", required=True)
a = p.parse_args()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
u.init_ulysses(cfg_parallel=a.cfg_parallel)
torch.manual_seed(2026)
model = (
    GAEFlowTemporal(
        input_size=8,
        in_channels=4,
        hidden_size=[128, 256],
        depth=[2, 2],
        num_heads=[16, 16],
        base_model_depth=0,
        use_rope=True,
        use_qknorm=True,
        use_rmsnorm=True,
        use_pos_embed=False,
        use_cross_attn=True,
        text_embed_dim=32,
        cross_attn_kdim=128,
        guidance_cross_attn=True,
        temporal_band_frac=0.5,
    )
    .cuda()
    .eval()
)
with torch.no_grad():
    model.final_layer.linear.weight.normal_(0, 0.025)
    model.base_final_layer.linear.weight.normal_(0, 0.025)
    for block in model.blocks:
        block.adaLN_modulation[-1].weight.normal_(0, 0.01)
        block.adaLN_modulation[-1].bias.normal_(0, 0.01)
        block.cross_attn_gate.fill_(0.25)
        for name, param in block.attn.plucker_pe.named_parameters():
            if param.ndim > 1:
                param.normal_(0, 0.03)

x = torch.randn(6, 4, 4, 8, device="cuda")
ref = torch.randn(2, 4, 4, 8, device="cuda")
text = torch.randn(2, 9, 32, device="cuda")
plucker = torch.randn(2, 3 * 32, 6, device="cuda")
times = torch.full((6,), 0.63, device="cuda")
results = []
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for conditional in (True, False):
        kw = dict(
            z_ref_clean=ref if conditional else None,
            plucker_6d=plucker if conditional else None,
            ref_global=text,
            cond_num=1 if conditional else 0,
            view_frame_idx=torch.tensor([3, 8, 2], device="cuda"),
            return_dict=True,
        )
        # Serial reference uses the same initialized group and exact model weights.
        u._ENABLED = False
        expected = model(x, times, 3, **kw)
        u.activate()
        actual = model(x, times, 3, **kw)
        for key in ("main", "base"):
            error = actual[key].float() - expected[key].float()
            relative = float(error.norm() / expected[key].float().norm().clamp_min(1e-8))
            assert relative < 0.025, (conditional, key, relative)
            results.append(
                dict(
                    conditional=conditional,
                    output=key,
                    max_abs=float(error.abs().max()),
                    relative_l2=relative,
                )
            )
if a.cfg_parallel:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "eval"))
    from eval_generation import sample_v4_euler

    model.fast_inference = True
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sample_args = dict(
            plucker_6d=plucker[:1],
            ref_global=text[:1],
            cfg_uncond_ref_global=-text[:1],
            num_steps=2,
            cfg_scale=2.0,
            time_dist_shift=6.364,
            prediction="x",
        )
        u._ENABLED = False
        u._CFG_PARALLEL = False
        torch.manual_seed(73)
        reference = sample_v4_euler(model, ref[:1], 3, 1, **sample_args)
        u._CFG_PARALLEL = True
        u.activate()
        torch.manual_seed(73)
        actual = sample_v4_euler(model, ref[:1], 3, 1, **sample_args)
        error = actual.float() - reference.float()
        relative = float(error.norm() / reference.float().norm())
        assert relative < 0.025, relative
        results.append(
            dict(output="cfg_euler_2steps", relative_l2=relative, max_abs=float(error.abs().max()))
        )
if u.global_rank() == 0:
    Path(a.output).write_text(
        json.dumps(dict(sp=u.world_size(), cfg=a.cfg_parallel, results=results), indent=2)
    )
    print(json.dumps(results), flush=True)
torch.distributed.destroy_process_group()
