import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    multi layer perceptron
    :param dim: input dimension
    :param hidden_dim: hidden layer dimension
    :param dropout: dropout rate
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass
        :param x: input tensor
        :return: output tensor
        """
        return self.net(x)
