import math
from typing import Optional, cast

import torch

from xtorch import util


class MultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_heads: int = 3,
        mask_following_timesteps: bool = False,
    ) -> None:
        super().__init__()  # type: ignore

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim or hidden_dim
        self._num_heads = num_heads
        self._mask_following_timesteps = mask_following_timesteps
        self._query_linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(self._input_dim, self._hidden_dim)
                for _ in range(self._num_heads)
            ]
        )
        self._key_linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(self._input_dim, self._hidden_dim)
                for _ in range(self._num_heads)
            ]
        )
        self._value_linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(self._input_dim, self._hidden_dim)
                for _ in range(self._num_heads)
            ]
        )
        self._output_linear = torch.nn.Linear(
            self._num_heads * self._hidden_dim, self._output_dim
        )

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    @staticmethod
    def _attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.BoolTensor,
        mask: Optional[torch.Tensor] = None,
        mask_following_timesteps: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ==========
            Q: `torch.Tensor`
                Query tensor of shape (batch_size, sequence_length, embedding_size)
            K: `torch.Tensor`
                Key tensor of shape (batch_size, sequence_length, embedding_size)
            V: `torch.Tensor`
                Value tensor of shape (batch_size, sequence_length, embedding_size)
            mask: `torch.BoolTensor`
                Sequence mask tensor of shape (batch_size, sequence_length)
        """
        assert Q.size() == K.size() == V.size()
        batch_size, sequence_length, embedding_size = Q.size()

        if mask is None:
            mask = Q.new_ones((batch_size, sequence_length), dtype=torch.bool)

        # Shape: (batch_size, sequence_length, sequence_length)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(embedding_size)

        # Shape: (batch_size, sequence_length, sequence_length)
        attention_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        if mask_following_timesteps:
            attention_mask = torch.tril(attention_mask)
        attention_mask = cast(torch.BoolTensor, attention_mask)

        # Shape: (batch_size, sequence_length, sequence_length)
        normalized_attention_scores = util.masked_softmax(
            attention_scores,
            attention_mask,
            dim=2,
        )

        # Shape: (batch_size, sequence_length, embedding_size)
        ret = torch.bmm(normalized_attention_scores, V)
        return ret

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ==========
            Q: `torch.Tensor`
                Query tensor of shape (batch_size, sequence_length, embedding_size)
            K: `torch.Tensor`
                Key tensor of shape (batch_size, sequence_length, embedding_size)
            V: `torch.Tensor`
                Value tensor of shape (batch_size, sequence_length, embedding_size)
            mask: `torch.BoolTensor`
                Sequence mask tensor of shape (batch_size, sequence_length)
        """
        heads = []
        for query_linear, key_linear, value_linear in zip(
            self._query_linear_layers,
            self._key_linear_layers,
            self._value_linear_layers,
        ):
            # Shape: (batch_size, sequence_length, embedding_size)
            head = self._attention(
                query_linear(Q),
                key_linear(K),
                value_linear(V),
                mask=mask,
                mask_following_timesteps=self._mask_following_timesteps,
            )
            heads.append(head)

        out = self._output_linear(torch.cat(heads, dim=2))
        return cast(torch.Tensor, out)
