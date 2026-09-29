"""Ulysses sequence parallelism for GAE inference.

Each rank holds a shard of the token sequence. Before attention, an all-to-all
exchanges that shard for a subset of heads so every rank runs attention on the
full sequence. A second all-to-all restores the sequence shard. Weights stay
replicated. Enable with ``torchrun`` and ``--ulysses``; training DDP does not
turn this on.
"""
from __future__ import annotations

import inspect
import os

import torch
import torch.distributed as dist

_ENABLED = False
_GROUP = None  # Ulysses subgroup; None uses the default process group
_CFG_PARALLEL = False
_BRANCH = "cond"


def enabled() -> bool:
    return _ENABLED


def cfg_parallel_enabled() -> bool:
    return _CFG_PARALLEL


def branch() -> str:
    """``cond`` on the low half of ranks, ``uncond`` on the high half."""
    return _BRANCH


def global_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def rank() -> int:
    if not dist.is_initialized():
        return 0
    if _GROUP is not None:
        return dist.get_rank(group=_GROUP)
    return dist.get_rank()


def world_size() -> int:
    if not dist.is_initialized():
        return 1
    if _GROUP is not None:
        return dist.get_world_size(group=_GROUP)
    return dist.get_world_size()


def activate() -> None:
    """Turn Ulysses on after the process group already exists."""
    global _ENABLED
    if not dist.is_initialized():
        raise RuntimeError("init_process_group before activate()")
    _ENABLED = world_size() > 1


def init_ulysses(backend: str | None = None, cfg_parallel: bool = False) -> int:
    """Initialize the default process group from a torchrun launch.

    With ``cfg_parallel``, ranks ``[0, N/2)`` run the conditional branch and
    ``[N/2, N)`` run the unconditional branch. Ulysses collectives stay inside
    each half, so each branch uses ``N/2`` sequence-parallel ranks.
    """
    global _GROUP, _CFG_PARALLEL, _BRANCH
    if not dist.is_initialized():
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    world = dist.get_world_size()
    if cfg_parallel:
        if world < 2 or world % 2 != 0:
            raise RuntimeError(
                f"cfg parallel needs an even world size >= 2, got {world}"
            )
        half = world // 2
        cond_group = dist.new_group(ranks=list(range(half)))
        uncond_group = dist.new_group(ranks=list(range(half, world)))
        gr = dist.get_rank()
        if gr < half:
            _GROUP = cond_group
            _BRANCH = "cond"
        else:
            _GROUP = uncond_group
            _BRANCH = "uncond"
        _CFG_PARALLEL = True
    activate()
    if global_rank() == 0:
        if _CFG_PARALLEL:
            half = world // 2
            print(
                f"[ulysses] cfg parallel: cond ranks 0..{half - 1}, "
                f"uncond ranks {half}..{world - 1}, "
                f"sequence parallel {world_size()}",
                flush=True,
            )
        elif enabled():
            print(f"[ulysses] sequence parallel across {world_size()} ranks", flush=True)
    return world_size()


def _broadcast(tensor: torch.Tensor, group, group_src: int = 0) -> None:
    """Broadcast from ``group_src`` inside ``group``.

    ``src`` is a global rank even when ``group`` is set. ``group_src`` is the
    rank inside ``group`` and exists on newer PyTorch; torch 2.5 only has ``src``.
    """
    if group is None:
        dist.broadcast(tensor, src=group_src)
        return
    if "group_src" in inspect.signature(dist.broadcast).parameters:
        dist.broadcast(tensor, group=group, group_src=group_src)
    else:
        dist.broadcast(
            tensor, src=dist.get_global_rank(group, group_src), group=group,
        )


def broadcast_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Copy ``tensor`` from local rank 0 so this Ulysses group shares state."""
    if tensor is None or not enabled():
        return tensor
    tensor = tensor.contiguous()
    _broadcast(tensor, _GROUP, group_src=0)
    return tensor


def broadcast_world(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Copy ``tensor`` from global rank 0 to every rank, across CFG branches."""
    if tensor is None or not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    tensor = tensor.contiguous()
    dist.broadcast(tensor, src=0)
    return tensor


def exchange_cfg(local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trade one branch's prediction so every rank receives ``(cond, uncond)``."""
    if not _CFG_PARALLEL:
        raise RuntimeError("exchange_cfg requires cfg parallel")
    local = local.contiguous()
    half = dist.get_world_size() // 2
    if dist.get_rank() < half:
        cond = local
        uncond = torch.empty_like(local)
    else:
        cond = torch.empty_like(local)
        uncond = local
    dist.broadcast(cond, src=0)
    dist.broadcast(uncond, src=half)
    return cond, uncond


def shard_span(length: int) -> tuple[int, int, int]:
    """Return ``(start, local_length, padded_length)`` for this rank.

    ``padded_length`` is the next multiple of the world size. Ranks whose span
    runs past ``length`` own zero-padded tokens that attention must ignore.
    """
    world = world_size()
    padded = (length + world - 1) // world * world
    local = padded // world
    start = rank() * local
    return start, local, padded


def take_shard(full: torch.Tensor, start: int, local: int, length: int) -> torch.Tensor:
    """Slice ``full`` of shape ``(B, length, C)`` and pad to ``local``."""
    out = full.new_zeros(full.shape[0], local, full.shape[-1])
    n = max(0, min(length, start + local) - start)
    if n:
        out[:, :n] = full[:, start:start + n]
    return out


def gather_sequence(local: torch.Tensor) -> torch.Tensor:
    """All-gather a ``(B, S_local, C)`` shard along the sequence dimension."""
    world = world_size()
    if world == 1:
        return local
    bufs = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(bufs, local.contiguous(), group=_GROUP)
    return torch.cat(bufs, dim=1)


def _exchange(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x.contiguous(), group=_GROUP)
    return out


def seq_to_head(x: torch.Tensor) -> torch.Tensor:
    """``(B, S_local, H, D)`` to ``(B, S, H_local, D)``."""
    batch, seq_local, heads, dim = x.shape
    world = world_size()
    if heads % world != 0:
        raise RuntimeError(
            f"Ulysses needs num_heads ({heads}) divisible by world size ({world})"
        )
    heads_local = heads // world
    packed = (
        x.reshape(batch, seq_local, world, heads_local, dim)
        .permute(2, 1, 0, 3, 4)
        .contiguous()
    )
    exchanged = _exchange(packed)
    return exchanged.flatten(0, 1).permute(1, 0, 2, 3).contiguous()


def head_to_seq(x: torch.Tensor) -> torch.Tensor:
    """``(B, S, H_local, D)`` to ``(B, S_local, H, D)``."""
    batch, seq, heads_local, dim = x.shape
    world = world_size()
    if seq % world != 0:
        raise RuntimeError(
            f"Ulysses sequence {seq} is not divisible by world size {world}"
        )
    seq_local = seq // world
    packed = (
        x.reshape(batch, world, seq_local, heads_local, dim)
        .permute(1, 3, 0, 2, 4)
        .contiguous()
    )
    exchanged = _exchange(packed)
    return exchanged.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
