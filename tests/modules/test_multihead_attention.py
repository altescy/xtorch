import torch

from xtorch.modules.multihead_attention import MultiheadAttention


def test_multihead_attention_initilalization() -> None:
    net = MultiheadAttention(
        input_dim=16,
        hidden_dim=8,
    )
    assert net.get_input_dim() == 16
    assert net.get_output_dim() == 8

    net = MultiheadAttention(
        input_dim=16,
        hidden_dim=8,
        output_dim=4,
    )
    assert net.get_input_dim() == 16
    assert net.get_output_dim() == 4


def test_multihead_attention_forward() -> None:
    net = MultiheadAttention(
        input_dim=16,
        hidden_dim=8,
        output_dim=4,
    )
    x = torch.randn((3, 10, 16))
    y = net(x, x, x)
    assert y.size() == (3, 10, 4)
