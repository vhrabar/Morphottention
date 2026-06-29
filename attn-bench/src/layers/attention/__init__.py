import os

if os.environ.get("ATTN_MOROPHO", "0") == "1":
    from .morpho_attention import MultiHeadSelfAttention
else:
    from .default import MultiHeadSelfAttention

__all__ = ["MultiHeadSelfAttention"]
