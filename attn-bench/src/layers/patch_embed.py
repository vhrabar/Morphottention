import torch
from torch import nn


class PatchEmbeder(nn.Module):
    """
    Patch Embeder class for ViT
    :param image_size: input image size
    :param patch_size: patch size
    :param channels: number of channels
    :param dim: patch dimension
    """

    def __init__(self, image_size: int, patch_size: int, channels: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the PatchEmbedder class
        :param x: input tensor (B C H W)
        :return: patch embedded tensor (B, num_patches, dim)
        """
        return self.proj(x).flatten(2).transpose(1, 2)
