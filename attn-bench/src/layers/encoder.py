from torch import nn

from .attention import MultiHeadSelfAttention
from .mlp import MLP


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_cls: type[nn.Module] = MultiHeadSelfAttention,
        attention_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        attention_kwargs = attention_kwargs or {}
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attention_cls(
            dim,
            num_heads,
            dropout=dropout,
            **attention_kwargs,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim=hidden_dim, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_bias=attn_bias)
        x = x + self.drop(attn_out)

        # MLP with residual
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x
