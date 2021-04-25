import pytest
import torch

from xtorch.exceptions import ConfigurationError
from xtorch.modules.activations import ReLU
from xtorch.modules.feedforward import FeedForward


def test_feedforward_initialization() -> None:
    net = FeedForward(
        input_dim=16,
        hidden_dims=8,
        num_layers=3,
        activations=ReLU(),
    )
    assert net.get_input_dim() == 16
    assert net.get_output_dim() == 8

    net = FeedForward(
        input_dim=16,
        hidden_dims=[8, 4],
        num_layers=2,
        activations=ReLU(),
    )
    assert net.get_input_dim() == 16
    assert net.get_output_dim() == 4

    with pytest.raises(ConfigurationError):
        # The length of `hidden_dims` is not equal to `num_layers`.
        FeedForward(
            input_dim=16,
            hidden_dims=[8, 4],
            num_layers=3,
            activations=ReLU(),
        )


def test_feedforward_forward() -> None:
    net = FeedForward(
        input_dim=16,
        hidden_dims=8,
        num_layers=3,
        activations=ReLU(),
        dropout=0.5,
    )

    x = torch.randn((4, 16))
    y = net(x)
    assert y.size() == (4, net.get_output_dim())
