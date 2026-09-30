"""Scene descriptions for uploaded images, and GPU picking for side jobs.

The caption model (Qwen2.5-VL-7B-Instruct by default, ~16 GB in bf16) is
loaded once on the GPU with the most free memory and reused. Bundled example
images keep their curated ``.txt`` descriptions.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "scenes"
CAPTION_MODEL = os.environ.get("CAPTION_MODEL") or "Qwen/Qwen2.5-VL-7B-Instruct"
INSTRUCTION = (
    "Describe this photograph as one English paragraph for a video-generation prompt. "
    "Cover the place, objects, lighting, and atmosphere that are actually visible. "
    "Do not mention a camera, an image, or a prompt, and do not invent motion."
)

_LOCK = threading.Lock()
_model = None
_example_hashes: dict[str, Path] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def known_caption(image: str) -> str | None:
    """The bundled description when ``image`` is (a copy of) an example image."""
    global _example_hashes
    path = Path(image)
    candidates = [path.with_suffix(".txt"), EXAMPLES / f"{path.stem}.txt"]
    if _example_hashes is None:
        _example_hashes = {_sha256(p): p for p in EXAMPLES.glob("*.jpg")}
    try:
        match = _example_hashes.get(_sha256(path))
    except OSError:
        match = None
    if match is not None:
        candidates.insert(0, match.with_suffix(".txt"))
    for text_path in candidates:
        if text_path.is_file() and (text := text_path.read_text(encoding="utf-8").strip()):
            return text
    return None


def least_used_gpu() -> tuple[int, int] | None:
    """(physical index, torch index) of the visible GPU with the most free memory."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    free = {int(i): float(m) for i, m in (line.split(",") for line in out.strip().splitlines())}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    order = [int(v) for v in visible.split(",") if v.strip().isdigit()] if visible else sorted(free)
    order = [gpu for gpu in order if gpu in free]
    if not order:
        return None
    physical = max(order, key=lambda gpu: free[gpu])
    return physical, order.index(physical)


def side_job_env(env: dict[str, str]) -> dict[str, str]:
    """Environment for a single-GPU helper process launched from a torchrun rank."""
    env = dict(env)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
                "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID"):
        env.pop(key, None)
    gpu = least_used_gpu()
    if gpu is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu[0])
    return env


def load():
    """Load the caption model once; returns (model, processor)."""
    global _model
    if _model is not None:
        return _model
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    major, minor = (int(v) for v in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    gpu = least_used_gpu() if torch.cuda.is_available() else None
    device = f"cuda:{gpu[1]}" if gpu is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = os.environ.get("CAPTION_DEVICE", device)
    print(f"[caption] loading {CAPTION_MODEL} on {device}", flush=True)
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CAPTION_MODEL, attn_implementation="sdpa", **{dtype_key: torch.bfloat16},
    ).to(device).eval()
    _model = (model, processor)
    return _model


def warm() -> None:
    try:
        with _LOCK:
            load()
    except Exception as exc:  # noqa: BLE001 - captions are optional
        print(f"[caption] warm-up failed: {type(exc).__name__}: {exc}", flush=True)


def stream(image: str, max_new_tokens: int = 160):
    """Yield the description as it is generated (whitespace-normalized)."""
    known = known_caption(image)
    if known:
        yield known
        return
    import torch
    from PIL import Image
    from transformers import TextIteratorStreamer

    errors: list[Exception] = []
    text = ""
    requested = time.perf_counter()
    with _LOCK:
        model, processor = load()
        started = time.perf_counter()
        picture = Image.open(image).convert("RGB")
        picture.thumbnail((768, 768))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]
        chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[chat], images=[picture], return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def run() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False)
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
                errors.append(exc)
                streamer.end()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        first = None
        for piece in streamer:
            first = first or time.perf_counter()
            text += piece
            yield " ".join(text.split())
        worker.join()
        done = time.perf_counter()
        tokens = len(processor.tokenizer(text)["input_ids"]) if text else 0
        print(f"[caption] {model.device}: waited {started - requested:.1f}s, "
              f"first text {(first or done) - started:.1f}s, "
              f"{tokens} tokens in {done - (first or done):.1f}s", flush=True)
    if errors:
        raise RuntimeError(f"{type(errors[0]).__name__}: {errors[0]}")
    if not text.strip():
        raise RuntimeError("the caption model returned an empty description")
