# =============================================================================
# models/vision_transformer.py
# Full Vision Transformer (ViT) implementation in PyTorch.
#
# Architecture follows the original paper:
#   "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale"
#   Dosovitskiy et al., 2020  (https://arxiv.org/abs/2010.11929)
#
# Components:
#   1. PatchEmbedding  – splits image into patches and projects to embed_dim
#   2. TransformerBlock – MHSA + FFN with residual connections & LayerNorm
#   3. VisionTransformer – full model with CLS token & classification head
# =============================================================================

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    IMAGE_SIZE, PATCH_SIZE, IN_CHANNELS, NUM_PATCHES,
    EMBED_DIM, NUM_HEADS, NUM_LAYERS, MLP_DIM,
    DROPOUT, NUM_CLASSES
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PATCH EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches and linearly embeds each patch.

    Uses a Conv2d layer with kernel_size=patch_size and stride=patch_size,
    which is mathematically equivalent to flattening patches and multiplying
    by a learned weight matrix – but much more efficient.

    Input  : (B, C, H, W)
    Output : (B, N, embed_dim)   where N = (H*W) / patch_size²
    """

    def __init__(
        self,
        image_size:  int = IMAGE_SIZE,
        patch_size:  int = PATCH_SIZE,
        in_channels: int = IN_CHANNELS,
        embed_dim:   int = EMBED_DIM,
    ) -> None:
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Convolutional projection: one convolution replaces patch-flattening + FC
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.projection(x)      # → (B, embed_dim, H/P, W/P)
        x = x.flatten(2)            # → (B, embed_dim, N)
        x = x.transpose(1, 2)       # → (B, N, embed_dim)
        x = self.norm(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTI-HEAD SELF-ATTENTION
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention.

    Splits embed_dim into num_heads heads, computes attention in parallel,
    and concatenates the results before a final linear projection.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        dropout:   float = DROPOUT,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, \
            "embed_dim must be divisible by num_heads"

        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5   # 1/√d_k

        # Project inputs to Q, K, V in one shot (3× embed_dim output)
        self.qkv        = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.attn_drop  = nn.Dropout(dropout)
        self.proj_drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)                          # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)           # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)                    # each: (B, heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        x = (attn @ v)                              # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)      # (B, N, embed_dim)
        x = self.proj_drop(self.projection(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEED-FORWARD NETWORK (MLP BLOCK)
# ─────────────────────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation used inside each Transformer block.

    Expands embed_dim → mlp_dim → embed_dim with residual connections
    applied *outside* this module (in TransformerBlock).
    """

    def __init__(
        self,
        embed_dim: int   = EMBED_DIM,
        mlp_dim:   int   = MLP_DIM,
        dropout:   float = DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRANSFORMER ENCODER BLOCK
# ─────────────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block.

    Pre-LayerNorm variant (more stable training):
        x = x + MHSA(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(
        self,
        embed_dim: int   = EMBED_DIM,
        num_heads: int   = NUM_HEADS,
        mlp_dim:   int   = MLP_DIM,
        dropout:   float = DROPOUT,
    ) -> None:
        super().__init__()
        self.norm1   = nn.LayerNorm(embed_dim)
        self.attn    = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2   = nn.LayerNorm(embed_dim)
        self.ff      = FeedForward(embed_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection around self-attention
        x = x + self.attn(self.norm1(x))
        # Residual connection around feed-forward
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISION TRANSFORMER (FULL MODEL)
# ─────────────────────────────────────────────────────────────────────────────
class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.

    Pipeline:
        Image → Patch Embedding → [CLS] concat → Positional Encoding
              → N × TransformerBlock → LayerNorm → CLS token → MLP Head → logits
    """

    def __init__(
        self,
        image_size:  int   = IMAGE_SIZE,
        patch_size:  int   = PATCH_SIZE,
        in_channels: int   = IN_CHANNELS,
        num_classes: int   = NUM_CLASSES,
        embed_dim:   int   = EMBED_DIM,
        num_heads:   int   = NUM_HEADS,
        num_layers:  int   = NUM_LAYERS,
        mlp_dim:     int   = MLP_DIM,
        dropout:     float = DROPOUT,
    ) -> None:
        super().__init__()

        num_patches = (image_size // patch_size) ** 2

        # ── Patch Embedding ──────────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

        # ── [CLS] Token ──────────────────────────────────────────────────────
        # Learnable classification token prepended to patch sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Positional Encoding ──────────────────────────────────────────────
        # +1 for the CLS token position
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_dropout = nn.Dropout(dropout)

        # ── Transformer Encoder ──────────────────────────────────────────────
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        # ── Final LayerNorm & Classification Head ────────────────────────────
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, num_classes),
        )

        # ── Weight Initialisation (following original ViT paper) ─────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier / trunc-normal initialisation for stable training."""
        nn.init.trunc_normal_(self.cls_token,     std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 1. Patch embedding: (B, C, H, W) → (B, N, embed_dim)
        x = self.patch_embed(x)

        # 2. Prepend CLS token: (B, N+1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 3. Add positional encoding
        x = x + self.pos_embedding
        x = self.pos_dropout(x)

        # 4. Transformer encoder blocks
        x = self.transformer(x)

        # 5. Final layer norm
        x = self.norm(x)

        # 6. Extract CLS token and classify
        cls_output = x[:, 0]          # (B, embed_dim)
        logits     = self.head(cls_output)    # (B, num_classes)
        return logits

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Convenience factory for a smaller ViT-Tiny (faster training / CPU) ─────
def vit_tiny(num_classes: int = NUM_CLASSES) -> VisionTransformer:
    """ViT-Tiny: suitable for quick experiments on CPU."""
    return VisionTransformer(
        embed_dim=192, num_heads=3, num_layers=12,
        mlp_dim=768, num_classes=num_classes
    )


def vit_small(num_classes: int = NUM_CLASSES) -> VisionTransformer:
    """ViT-Small: balanced speed and accuracy."""
    return VisionTransformer(
        embed_dim=384, num_heads=6, num_layers=12,
        mlp_dim=1536, num_classes=num_classes
    )


def vit_base(num_classes: int = NUM_CLASSES) -> VisionTransformer:
    """ViT-Base: full model as described in the original paper."""
    return VisionTransformer(
        embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        mlp_dim=MLP_DIM, num_classes=num_classes
    )


if __name__ == "__main__":
    model = vit_tiny()
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out   = model(dummy)
    print(f"Output shape      : {out.shape}")          # (2, 10)
    print(f"Trainable params  : {model.get_num_params():,}")
