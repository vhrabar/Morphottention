import torch

def forward(
    X: torch.Tensor,
    W_phi: torch.Tensor,
    gate_q: torch.Tensor,
    gate_k: torch.Tensor,
    W_V: torch.Tensor,
    H: int,
    cube_m: int,
    scale: float,
    causal: bool,
) -> list[torch.Tensor]: ...

def backward(grad_out: torch.Tensor, X: torch.Tensor) -> list[torch.Tensor]: ...
