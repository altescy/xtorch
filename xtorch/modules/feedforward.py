from typing import List, Union

import torch

from xtorch.exceptions import ConfigurationError
from xtorch.modules.activations import Activation


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_dims: Union[int, List[int]],
        activations: Union[Activation, List[Activation]],
        dropout: Union[float, List[float]] = 0.0,
    ) -> None:
        super().__init__()  # type: ignore

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers
        if not isinstance(activations, list):
            activations = [activations] * num_layers
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers

        if len(hidden_dims) != num_layers:
            raise ConfigurationError(
                f"len(hidden_dims) ({len(hidden_dims)}) "
                f"!= num_layers ({num_layers})"
            )
        if len(activations) != num_layers:
            raise ConfigurationError(
                f"len(activations) ({len(activations)}) "
                f"!= num_layers ({num_layers})"
            )
        if len(dropout) != num_layers:
            raise ConfigurationError(
                f"len(dropout) ({len(dropout)}) " f"!= num_layers ({num_layers})"
            )

        input_dims = [input_dim] + hidden_dims[:-1]
        self._linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(input_dim, output_dim)
                for input_dim, output_dim in zip(input_dims, hidden_dims)
            ]
        )
        self._activations = torch.nn.ModuleList(activations)
        self._dropout = torch.nn.ModuleList(
            [torch.nn.Dropout(p=value) for value in dropout]
        )
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def get_input_dim(self) -> int:
        return self.input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer, activation, dropout in zip(
            self._linear_layers,
            self._activations,
            self._dropout,
        ):
            output = dropout(activation(layer(output)))
        return output
