"""Inference-only optimizations preserve broadcast and per-query semantics."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stage2.models.DDT import DDTGate, DDTModulate
from stage2.models.ddt_head import LightningDDTBlockV3


def test_single_condition_broadcast_is_exact():
    torch.manual_seed(1)
    x = torch.randn(2, 13, 16)
    shift = torch.randn(2, 1, 16)
    scale = torch.randn_like(shift)
    with torch.no_grad():
        assert torch.equal(
            DDTModulate(x, shift, scale),
            x * (1 + scale.repeat_interleave(13, 1)) + shift.repeat_interleave(13, 1),
        )
        assert torch.equal(DDTGate(x, scale), x * scale.repeat_interleave(13, 1))


def test_segmented_condition_and_gradients_still_work():
    x = torch.randn(2, 12, 16, requires_grad=True)
    shift = torch.randn(2, 3, 16, requires_grad=True)
    scale = torch.randn_like(shift)
    expected = x * (1 + scale.repeat_interleave(4, 1)) + shift.repeat_interleave(4, 1)
    actual = DDTModulate(x, shift, scale)
    assert torch.equal(actual, expected)
    a = torch.autograd.grad(actual.sum(), (x, shift), retain_graph=True)
    b = torch.autograd.grad(expected.sum(), (x, shift))
    for before, after in zip(a, b):
        assert torch.equal(before, after)


@pytest.mark.parametrize("batch,views", [(1, 81), (2, 3)])
def test_folded_text_attention_preserves_batch_boundaries(batch, views):
    torch.manual_seed(5)
    block = LightningDDTBlockV3(64, 8, use_cross_attn=True, cross_attn_kdim=32).eval()
    query = torch.randn(batch * views, 7, 64)
    context = torch.randn(batch, 9, 32)
    with torch.no_grad():
        block.fold_text_queries = False
        before = block._text_attention(query, context)
        block.fold_text_queries = True
        after = block._text_attention(query, context)
    torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-5)
