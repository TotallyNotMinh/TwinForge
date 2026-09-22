import torch
from torch import nn
from models.transformer_block import TransformerBlock
import torch.nn.functional as F

class TemporalHead(nn.Module):
    def __init__(self, num_heads, max_frames, size=(378, 504), embed_dim=128):
        super().__init__()
        self.size = size
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.proj4 = nn.Sequential(nn.Conv2d(1024, embed_dim, kernel_size=1, bias=False), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.proj5 = nn.Sequential(nn.Conv2d(2048, embed_dim, kernel_size=1, bias=False), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        
        self.temporal_embed = nn.Parameter(data=torch.randn((1, max_frames, embed_dim)) * 0.02)

        self.transformer4 = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(3)])
        self.transformer5 = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(3)])

        self.gamma = nn.Parameter(torch.zeros(1)) # Scaling factor

    def forward(self, features, B, T):
        f4_proj = self.proj4(features["f4"])
        f5_proj = self.proj5(features["f5"])

        BT, C, H4, W4 = f4_proj.shape
        BT, C, H5, W5 = f5_proj.shape

        f4_reshape = f4_proj.reshape(B, T, C, H4, W4) # (B * T, C, H, W) -> (B, T, C, H, W)
        f5_reshape = f5_proj.reshape(B, T, C, H5, W5)

        f4_perm = torch.permute(f4_reshape, (0, 3, 4, 1, 2)) # (B, T, C, H, W) -> (B, H, W, T, C)
        f5_perm = torch.permute(f5_reshape, (0, 3, 4, 1, 2))

        f4_flat = torch.flatten(f4_perm, start_dim=0, end_dim=2) # (B, H, W, T, C) -> (B * H * W, T, C) to compute attention on T dimension only
        f5_flat = torch.flatten(f5_perm, start_dim=0, end_dim=2)

        trans_f4 = self.transformer4(f4_flat + self.temporal_embed[:, :T, :]).reshape(B, H4, W4, T, self.embed_dim).permute(0, 3, 4, 1, 2).reshape(B * T, C, H4, W4) # Reshape output back to original shape
        trans_f5 = self.transformer5(f5_flat + self.temporal_embed[:, :T, :]).reshape(B, H5, W5, T, self.embed_dim).permute(0, 3, 4, 1, 2).reshape(B * T, C, H5, W5)

        target_size = features["f2"].shape[-2:]
        trans_f4_up = F.interpolate(trans_f4, size=target_size, mode="bilinear", align_corners=False)
        trans_f5_up = F.interpolate(trans_f5, size=target_size, mode="bilinear", align_corners=False)

        return trans_f4_up * self.gamma, trans_f5_up * self.gamma