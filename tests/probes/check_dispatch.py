"""Two-process CPU/Gloo protocol check; run with torchrun, no model downloads."""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.demo.distributed_engine import SamplerGroup

dist.init_process_group("gloo")
group = SamplerGroup(None, torch.device("cpu"))
for size in (3, 5):
    packet = {
        "seed": size * 13,
        "ref": torch.arange(size * 2, dtype=torch.float32).reshape(2, size),
        "text": torch.arange(size, dtype=torch.bfloat16),
        "view_frame_idx": torch.arange(size),
        "optional": None,
    }
    received = group._broadcast(packet if dist.get_rank() == 0 else None)
    assert received["seed"] == size * 13 and received["optional"] is None
    for name in ("ref", "text", "view_frame_idx"):
        assert torch.equal(received[name], packet[name])
if dist.get_rank() == 0:
    group.stop()
else:
    assert group._broadcast()["stop"]
dist.destroy_process_group()
print("DISPATCH_OK", os.environ["RANK"], flush=True)
