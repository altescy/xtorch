from typing import Optional, cast

import torch

from xtorch.modules.activations import ReLU
from xtorch.modules.feedforward import FeedForward
from xtorch.modules.multihead_attention import MultiheadAttention
from xtorch.modules.positional_encoders import (
    PositionalEncoder,
    TrigonometricPositionalEncoder,
)
from xtorch.modules.seq2seq_encoders.seq2seq_encoder import Seq2seqEncoder


class SelfAttentiveSeq2seqEncoder(Seq2seqEncoder):  # type: ignore
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 3,
        output_dim: Optional[int] = None,
        feedforward_dim: Optional[int] = None,
        feedforward_num_layers: int = 2,
        positional_encoder: Optional[PositionalEncoder] = None,
        mask_following_timesteps: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._output_dim = output_dim or hidden_dim
        self._feedforward_dim = feedforward_dim or self._hidden_dim
        self._feedforward_num_layers = feedforward_num_layers
        self._positional_encoder = (
            positional_encoder or TrigonometricPositionalEncoder()
        )

        self._input_linear_layer = torch.nn.Linear(self._input_dim, self._hidden_dim)
        self._multihead_attentions = torch.nn.ModuleList(
            [
                MultiheadAttention(
                    input_dim=self._hidden_dim,
                    hidden_dim=self._hidden_dim,
                    num_heads=self._num_heads,
                    mask_following_timesteps=mask_following_timesteps,
                )
                for _ in range(self._num_layers)
            ]
        )

        self._feedforward_layers = torch.nn.ModuleList(
            [
                FeedForward(
                    input_dim=self._hidden_dim,
                    hidden_dims=self._feedforward_dim,
                    num_layers=self._feedforward_num_layers,
                    activations=ReLU(),
                )
            ]
        )

        self._layer_norms_after_attention = torch.nn.ModuleList(
            [torch.nn.LayerNorm(self._hidden_dim) for _ in range(self._num_layers)]
        )
        self._layer_norms_after_feedforward = torch.nn.ModuleList(
            [torch.nn.LayerNorm(self._hidden_dim) for _ in range(self._num_layers)]
        )

        self._output_layer = torch.nn.Linear(
            self._feedforward_layers[-1].get_output_dim(), self._output_dim
        )

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ==========
        inputs: `torch.Tensor`
            Tensor of shape (batch_size, sequence_length, embedding_size).
        mask: `torch.BoolTensor`
            BoolTensor of shape (batch_size, sequence_length).

        Return
        ======
        output:
            Tensor of shape (batch_size, sequence_length, encoding_size).
        """
        h = self._input_linear_layer(inputs)

        if self._positional_encoder is not None:
            h = self._positional_encoder(h)

        for (
            multihead_attention,
            feedforward,
            layer_norm_after_attention,
            layer_norm_after_feedforward,
        ) in zip(
            self._multihead_attentions,
            self._feedforward_layers,
            self._layer_norms_after_attention,
            self._layer_norms_after_feedforward,
        ):
            h = multihead_attention(h, h, h, mask=mask) + h
            h = layer_norm_after_attention(h)
            h = feedforward(h) + h
            h = layer_norm_after_feedforward(h)

        output = self._output_layer(h)

        return cast(torch.Tensor, output)
