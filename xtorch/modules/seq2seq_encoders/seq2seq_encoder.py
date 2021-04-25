from typing import Optional

import torch


class Seq2seqEncoder(torch.nn.Module):
    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ==========
        inputs: `torch.Tensor`
            Tensor of shape (batch_size, sequence_length, embedding_size).
        mask: `torch.BoolTensor`, optional (default = None)
            BoolTensor of shape (batch_size, sequence_length).

        Return
        ======
        output:
            Tensor of shape (batch_size, sequence_length, encoding_size).
        """
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError
