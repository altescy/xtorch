import torch


class PositionalEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameter
        ========
        inputs: `torch.Tensor`
            Tensor of shape (batch_size, sequence_length, embedding_size).

        Return
        ======
        output: `torch.Tensor`
            Tensor of shape (batch_size, sequence_length, encoding_size).
        """
        raise NotImplementedError


class TrigonometricPositionalEncoder(PositionalEncoder):
    @staticmethod
    def forward(inputs: torch.Tensor) -> torch.Tensor:
        _batch_size, sequence_length, embedding_size = inputs.size()
        positions = (
            torch.arange(
                sequence_length,
                device=inputs.device,
            )
            .repeat(embedding_size, 1)
            .transpose(0, 1)
        )
        dims = torch.arange(
            1,
            embedding_size + 1,
            device=inputs.device,
        ).repeat(sequence_length, 1)

        encodings = inputs.new_zeros((sequence_length, embedding_size))
        odd_encoding = torch.cos(
            positions / torch.pow(10000, (dims - 1) / embedding_size)
        )
        even_encoding = torch.sin(positions / torch.pow(10000, dims / embedding_size))
        even_mask = (torch.arange(embedding_size) % 2).bool()
        encodings[:, even_mask] = even_encoding[:, even_mask]
        encodings[:, ~even_mask] = odd_encoding[:, ~even_mask]

        encodings = encodings.unsqueeze(0)
        return inputs + encodings
