"""Full policy graphs preserve masks, padding, values and failure detection."""
import pytest
import torch

from AlphaZero.model.cuda_graph import CudaGraphPolicy
from AlphaZero.model.network import PolicyValueNetwork


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_policy_graph_matches_eager_across_reused_padded_batches(precision):
    torch.manual_seed(73)
    model = PolicyValueNetwork(1, 8).cuda().eval().to(memory_format=torch.channels_last)
    graph = CudaGraphPolicy(model, precision, True, 8)
    for count in (3, 4, 1, 3, 8, 5):
        states = torch.randn(count, 131, 10, 9, device="cuda").contiguous(memory_format=torch.channels_last)
        masks = torch.rand(count, 8100, device="cuda") > .98
        masks[:, 0] = True
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                  enabled=precision == "bf16", cache_enabled=False):
            logits, values = model(states)
            expected = logits.float().masked_fill(~masks, -torch.inf).softmax(1)
        actual = graph(states, masks).clone()
        tolerance = 2e-3 if precision == "bf16" else 2e-6
        torch.testing.assert_close(actual[:, :-2], expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(actual[:, -2], values.float(), rtol=tolerance, atol=tolerance)
        assert (actual[:, -1] == 1).all()
        assert (actual[:, :-2][~masks] == 0).all()
        torch.testing.assert_close(actual[:, :-2].sum(1), torch.ones(count, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_policy_graph_reports_nonfinite_illegal_logits():
    class InvalidNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, device="cuda"))

        def forward(self, states):
            logits = torch.zeros(len(states), 8100, device="cuda") + self.weight
            logits[:, 1] = torch.nan
            return logits, torch.zeros(len(states), device="cuda")

    model = InvalidNetwork().eval()
    graph = CudaGraphPolicy(model)
    masks = torch.zeros(1, 8100, dtype=torch.bool, device="cuda")
    masks[:, 0] = True
    output = graph(torch.zeros(1, 131, 10, 9, device="cuda"), masks)
    assert output[0, 0] == 1
    assert output[0, -1] == 0
