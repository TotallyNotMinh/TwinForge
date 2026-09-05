import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from models.transformer_block import TransformerBlock
from models.encoder import ResNetEncoder
import torch.nn.functional as F

class PatchEmbeder(nn.Module):
    def __init__(self, c_in, tok_dim, patch_size):
        super().__init__()
        self.embedder = nn.Conv2d(c_in, tok_dim, kernel_size=patch_size, stride=patch_size) #Patches are independent, no overlap

    def forward(self, x):
        return self.embedder(x)

class CrossTaskRefinementBlock(nn.Module):
    def __init__(self, tok_dim, num_heads):
        super().__init__()
        self.depth_cross = TransformerBlock(tok_dim, num_heads)
        self.segment_cross = TransformerBlock(tok_dim, num_heads)

    def forward(self, depth_token, segment_token):
        d_out = self.depth_cross(depth_token, context=segment_token)
        s_out = self.segment_cross(segment_token, context=depth_token)
        return d_out, s_out

class SegmentDecoder(nn.Module):
    def __init__(self, tok_dim, num_labels, embed_dim=128):
        super().__init__()
        self.proj3 = nn.Sequential(nn.Conv2d(512 + tok_dim, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.proj2 = nn.Sequential(nn.Conv2d(256 + tok_dim, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, tok_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(tok_dim),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(tok_dim * 2 + 64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_labels, kernel_size=3, padding=1)
        )

    def forward(self, vit_segment_self, p5, p4, tokens, features):
        target_size = features["f1"].shape[-2:]

        p3 = F.interpolate(self.proj3(torch.cat([tokens["s3"], features["f3"]], dim=1)), size=target_size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.proj2(torch.cat([tokens["s2"], features["f2"]], dim=1)), size=target_size, mode="bilinear", align_corners=False)

        fused = self.fuse(torch.cat([p5, p4, p3, p2], dim=1))
        return self.out(torch.cat([vit_segment_self, fused, features["f1"]], dim=1))


class DepthDecoder(nn.Module):
    def __init__(self, tok_dim, min_depth=1e-3, max_depth=10.0, embed_dim=128):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.proj3 = nn.Sequential(nn.Conv2d(512 + tok_dim, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.proj2 = nn.Sequential(nn.Conv2d(256 + tok_dim, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, tok_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(tok_dim),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(tok_dim * 2 + 64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 1, kernel_size=3, padding=1)
        )
        nn.init.constant_(self.out[3].bias, -1.0986)

    def forward(self, vit_depth_self, p5, p4, tokens, features):
        target_size = features["f1"].shape[-2:]

        p3 = F.interpolate(self.proj3(torch.cat([tokens["d3"], features["f3"]], dim=1)), size=target_size, mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.proj2(torch.cat([tokens["d2"], features["f2"]], dim=1)), size=target_size, mode="bilinear", align_corners=False)

        fused = self.fuse(torch.cat([p5, p4, p3, p2], dim=1))
        out = self.out(torch.cat([vit_depth_self, fused, features["f1"]], dim=1))
        return self.min_depth + (self.max_depth - self.min_depth) * torch.sigmoid(out)


class MultiHeadDecoder(nn.Module):
    def __init__(self, num_labels, tok_dim, num_heads, embed_dim=128, freeze=False):
        super().__init__()

        self.tok_dim = tok_dim
        self.num_heads = num_heads

        self.flatten = nn.Flatten(start_dim=2)

        self.patch_embedder1 = PatchEmbeder(64, tok_dim, patch_size=6)
        self.patch_embedder2 = PatchEmbeder(256, tok_dim, patch_size=4)
        self.patch_embedder3 = PatchEmbeder(512, tok_dim, patch_size=4)
        self.patch_embedder4 = PatchEmbeder(1024, tok_dim, patch_size=3)
        self.patch_embedder5 = PatchEmbeder(2048, tok_dim, patch_size=3)

        self.pos_embed1 = nn.Parameter(torch.randn(1, tok_dim, 24, 32) * 0.02)
        self.pos_embed2 = nn.Parameter(torch.randn(1, tok_dim, 18, 24) * 0.02)
        self.pos_embed3 = nn.Parameter(torch.randn(1, tok_dim, 9, 12) * 0.02)
        self.pos_embed4 = nn.Parameter(torch.randn(1, tok_dim, 6, 8) * 0.02)
        self.pos_embed5 = nn.Parameter(torch.randn(1, tok_dim, 3, 4) * 0.02)
        self.level_embed = nn.Embedding(5, tok_dim)

        # Asymmetric multi-scale feature transformers:
        # Scales 1 & 2: 1 block (high-res spatial detail with low compute)
        # Scale 3: 2 blocks (mid-level part composition)
        # Scales 4 & 5: 3 blocks (deep global 3D/scene context)
        self.transformer1 = TransformerBlock(tok_dim, num_heads)
        self.transformer2 = TransformerBlock(tok_dim, num_heads)
        self.transformer3 = nn.Sequential(*[TransformerBlock(tok_dim, num_heads) for _ in range(2)])
        self.transformer4 = nn.Sequential(*[TransformerBlock(tok_dim, num_heads) for _ in range(3)])
        self.transformer5 = nn.Sequential(*[TransformerBlock(tok_dim, num_heads) for _ in range(3)])

        self.fuse_transformer_depth_self = TransformerBlock(tok_dim, num_heads)
        self.fuse_transformer_segment_self = TransformerBlock(tok_dim, num_heads)

        # Multi-stage mutual cross-task refinement (2 iterations)
        self.cross_layers = nn.ModuleList([
            CrossTaskRefinementBlock(tok_dim, num_heads) for _ in range(2)
        ])

        # Shared high-level semantic & context decoders (dec5 and dec4)
        self.shared_proj5 = nn.Sequential(
            nn.Conv2d(2048 + tok_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.shared_proj4 = nn.Sequential(
            nn.Conv2d(1024 + tok_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.segment_dec = SegmentDecoder(tok_dim, num_labels, embed_dim=embed_dim)
        self.depth_dec = DepthDecoder(tok_dim, embed_dim=embed_dim)

    def _get_pos_embed(self, pos_param, h, w):
        if pos_param.shape[-2:] == (h, w):
            return pos_param
        return F.interpolate(pos_param, size=(h, w), mode="bicubic", align_corners=False)

    def forward(self, features):
        B = features["f1"].shape[0]

        patch1 = self.patch_embedder1(features["f1"])
        h1, w1 = patch1.shape[-2:]
        pe1 = self._get_pos_embed(self.pos_embed1, h1, w1)
        tok1 = (patch1 + pe1).flatten(2).transpose(1, 2) + self.level_embed.weight[0]
        l1 = h1 * w1

        patch2 = self.patch_embedder2(features["f2"])
        h2, w2 = patch2.shape[-2:]
        pe2 = self._get_pos_embed(self.pos_embed2, h2, w2)
        tok2 = (patch2 + pe2).flatten(2).transpose(1, 2) + self.level_embed.weight[1]
        l2 = h2 * w2

        patch3 = self.patch_embedder3(features["f3"])
        h3, w3 = patch3.shape[-2:]
        pe3 = self._get_pos_embed(self.pos_embed3, h3, w3)
        tok3 = (patch3 + pe3).flatten(2).transpose(1, 2) + self.level_embed.weight[2]
        l3 = h3 * w3

        patch4 = self.patch_embedder4(features["f4"])
        h4, w4 = patch4.shape[-2:]
        pe4 = self._get_pos_embed(self.pos_embed4, h4, w4)
        tok4 = (patch4 + pe4).flatten(2).transpose(1, 2) + self.level_embed.weight[3]
        l4 = h4 * w4

        patch5 = self.patch_embedder5(features["f5"])
        h5, w5 = patch5.shape[-2:]
        pe5 = self._get_pos_embed(self.pos_embed5, h5, w5)
        tok5 = (patch5 + pe5).flatten(2).transpose(1, 2) + self.level_embed.weight[4]
        l5 = h5 * w5

        trans_output1 = self.transformer1(tok1)
        trans_output2 = self.transformer2(tok2)
        trans_output3 = self.transformer3(tok3)
        trans_output4 = self.transformer4(tok4)
        trans_output5 = self.transformer5(tok5)

        joint_multi_layer_token = torch.concat([trans_output1, trans_output2, trans_output3, trans_output4, trans_output5], dim=1)

        # Self Attention
        depth_token_self = self.fuse_transformer_depth_self(joint_multi_layer_token)
        segment_token_self = self.fuse_transformer_segment_self(joint_multi_layer_token)

        # Multi-stage mutual cross-task refinement
        depth_token_cross = depth_token_self
        segment_token_cross = segment_token_self
        for cross_layer in self.cross_layers:
            depth_token_cross, segment_token_cross = cross_layer(depth_token_cross, segment_token_cross)

        # Dynamic Token Slices
        s1_idx = 0
        s2_idx = s1_idx + l1
        s3_idx = s2_idx + l2
        s4_idx = s3_idx + l3
        s5_idx = s4_idx + l4

        # --- Scale 1 from Self-Attention (for Task Heads) ---
        d1 = depth_token_self[:, s1_idx : s1_idx + l1, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h1, w1)
        s1 = segment_token_self[:, s1_idx : s1_idx + l1, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h1, w1)

        vit_depth_self = F.interpolate(d1, size=features["f1"].shape[-2:], mode="bilinear", align_corners=False)
        vit_segment_self = F.interpolate(s1, size=features["f1"].shape[-2:], mode="bilinear", align_corners=False)

        # --- Scales 5 down to 2 from Cross-Attention (for Gated Decoder) ---
        d5 = depth_token_cross[:, s5_idx : s5_idx + l5, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h5, w5)
        s5 = segment_token_cross[:, s5_idx : s5_idx + l5, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h5, w5)

        d4 = depth_token_cross[:, s4_idx : s4_idx + l4, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h4, w4)
        s4 = segment_token_cross[:, s4_idx : s4_idx + l4, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h4, w4)

        d3 = depth_token_cross[:, s3_idx : s3_idx + l3, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h3, w3)
        s3 = segment_token_cross[:, s3_idx : s3_idx + l3, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h3, w3)

        d2 = depth_token_cross[:, s2_idx : s2_idx + l2, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h2, w2)
        s2 = segment_token_cross[:, s2_idx : s2_idx + l2, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h2, w2)

        d5 = F.interpolate(d5, size=features["f5"].shape[-2:], mode="bilinear", align_corners=False)
        s5 = F.interpolate(s5, size=features["f5"].shape[-2:], mode="bilinear", align_corners=False)

        d4 = F.interpolate(d4, size=features["f4"].shape[-2:], mode="bilinear", align_corners=False)
        s4 = F.interpolate(s4, size=features["f4"].shape[-2:], mode="bilinear", align_corners=False)

        d3 = F.interpolate(d3, size=features["f3"].shape[-2:], mode="bilinear", align_corners=False)
        s3 = F.interpolate(s3, size=features["f3"].shape[-2:], mode="bilinear", align_corners=False)

        d2 = F.interpolate(d2, size=features["f2"].shape[-2:], mode="bilinear", align_corners=False)
        s2 = F.interpolate(s2, size=features["f2"].shape[-2:], mode="bilinear", align_corners=False)

        target_size = features["f1"].shape[-2:]
        shared_p5 = F.interpolate(self.shared_proj5(torch.cat([d5, s5, features["f5"]], dim=1)), size=target_size, mode="bilinear", align_corners=False)
        shared_p4 = F.interpolate(self.shared_proj4(torch.cat([d4, s4, features["f4"]], dim=1)), size=target_size, mode="bilinear", align_corners=False)

        depth_tokens = {"d3": d3, "d2": d2}
        segment_tokens = {"s3": s3, "s2": s2}

        depth_logits = self.depth_dec(vit_depth_self, shared_p5, shared_p4, depth_tokens, features)
        segment_logits = self.segment_dec(vit_segment_self, shared_p5, shared_p4, segment_tokens, features)

        return depth_logits, segment_logits

if __name__ == "__main__":
    model = MultiHeadDecoder(tok_dim=128, num_heads=4, num_labels=40)
    im = torch.randn([1, 3, 288, 384])
    encoder = ResNetEncoder()
    features = encoder(im)
    model(features)
