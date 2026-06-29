import torch
import torch.nn as nn

from src.configs import ModelConfig
from src.layers import EncoderBlock, PatchEmbeder


class ViT(nn.Module):
    def __init__(self, params: ModelConfig) -> None:
        super().__init__()
        self.patch_embed = PatchEmbeder(params.image_size, params.patch_size, params.in_chans, params.embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, params.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, params.num_patches + 1, params.embed_dim))
        self.pos_drop = nn.Dropout(params.dropout)

        self.blocks = nn.Sequential(*[self._make_block(params) for idx in range(params.depth)])

        self.norm = nn.LayerNorm(params.embed_dim)
        self.head = nn.Linear(params.embed_dim, params.num_classes)

    @staticmethod
    def _make_block(params: ModelConfig) -> nn.Module:
        return EncoderBlock(
            params.embed_dim,
            params.num_heads,
            params.mlp_ratio,
            params.dropout,
        )

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed the input image into patch embeddings.
        :param x: input tensor ( B, C, H, W )
        :return: patch embeddings ( B, num_patches, embed_dim )
        """
        B = x.shape[0]

        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)

        return self.norm(x)

    def _classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ViT model.
        :param x: input tensor ( B, C, H, W )
        :return: output logits ( B, num_classes )
        """
        # patch & embed
        x = self._embed(x)

        # encoder block
        x = self._encode(x)

        # classifier
        return self._classify(x)
