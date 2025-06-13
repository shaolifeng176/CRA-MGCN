import torch
import torch.nn as nn
from CRMSA import CrossRegionAttntion


class CrossRegionAttentionWrapper(nn.Module):
    def __init__(self, in_features, n_heads=4, dropout=0.1, region_size=8):
        super().__init__()
        self.crmsa = CrossRegionAttntion(
            dim=in_features,
            num_heads=n_heads,
            region_size=region_size,
            attn_drop=dropout,
            drop=dropout
        )

    def forward(self, x, mask=None):
        # Reshape input for CR-MSA: [batch, seq_len, features] -> [batch*seq_len, 1, features]
        B, N, C = x.shape
        x = x.view(B * N, 1, C)

        # Apply CR-MSA
        x = self.crmsa(x)

        # Reshape back: [batch*seq_len, 1, features] -> [batch, seq_len, features]
        x = x.view(B, N, C)
        return x, None  # Return None for attention weights to maintain compatibility