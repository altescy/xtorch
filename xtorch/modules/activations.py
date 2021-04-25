import torch


class Activation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearActivation(Activation):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class ReLU(Activation):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(inputs)


class LeakyReLU(Activation):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(inputs)


class ELU(Activation):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(inputs)


class Sigmoid(Activation):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(inputs)


class Tanh(Activation):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(inputs)


class Swish(Activation):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * torch.nn.functional.sigmoid(inputs)
