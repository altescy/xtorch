import torch

from xtorch.modules.seq2seq_encoders.self_attentive_encoder import (
    SelfAttentiveSeq2seqEncoder,
)


def test_self_attentive_seq2seq_encoder():
    net = SelfAttentiveSeq2seqEncoder(
        input_dim=16,
        hidden_dim=8,
        num_heads=3,
        num_layers=2,
    )
    x = torch.randn((3, 5, 16))
    y = net(x)
    assert y.size() == (3, 5, net.get_output_dim())
