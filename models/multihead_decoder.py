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

class SegmentHead(nn.Module):
    def __init__(self, tok_dim, num_labels):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(tok_dim * 2 + 64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_labels, kernel_size=3, padding=1)
        )

    def forward(self, token, decode_feature, encode_feature):
        return self.out(torch.concat([token, encode_feature, decode_feature], dim=1)) 


class DepthHead(nn.Module):
    def __init__(self, tok_dim):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(tok_dim * 2 + 64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 1, kernel_size=3, padding=1)
        )

    def forward(self, token, decode_feature, encode_feature):
        x = self.out(torch.concat([token, encode_feature, decode_feature], dim=1)) 
        return torch.clamp(x, min=1e-3, max=10.0)


class MultiHeadDecoder(nn.Module):
    def __init__(self, num_labels, tok_dim, num_heads, freeze=False):
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

        self.transformer1 = TransformerBlock(tok_dim, num_heads)
        self.transformer2 = TransformerBlock(tok_dim, num_heads)
        self.transformer3 = TransformerBlock(tok_dim, num_heads)
        self.transformer4 = TransformerBlock(tok_dim, num_heads)
        self.transformer5 = TransformerBlock(tok_dim, num_heads)

        self.fuse_transformer_depth_self = TransformerBlock(tok_dim, num_heads)
        self.fuse_transformer_segment_self = TransformerBlock(tok_dim, num_heads)

        self.fuse_transformer_depth_cross = TransformerBlock(tok_dim, num_heads)
        self.fuse_transformer_segment_cross = TransformerBlock(tok_dim, num_heads)

        self.fuse_proj_depth = nn.Sequential(
            nn.Conv2d(tok_dim, tok_dim, kernel_size=1),
            nn.BatchNorm2d(tok_dim),
            nn.ReLU(inplace=True)
        )
        self.fuse_proj_segment = nn.Sequential(
            nn.Conv2d(tok_dim, tok_dim, kernel_size=1),
            nn.BatchNorm2d(tok_dim),
            nn.ReLU(inplace=True)
        )
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(tok_dim * 2, tok_dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.dec5 = nn.Sequential(
            nn.Conv2d(2048 + tok_dim, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.dec4 = nn.Sequential(
            nn.Conv2d(1024 * 2 + tok_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(512 * 2 + tok_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(256 * 2 + tok_dim, tok_dim, kernel_size=1),
            nn.BatchNorm2d(tok_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(tok_dim, tok_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(tok_dim),
            nn.ReLU(inplace=True)
        )

        self.segment_head = SegmentHead(tok_dim, num_labels)
        self.depth_head = DepthHead(tok_dim)

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

        # Cross Attention
        depth_token_cross = self.fuse_transformer_depth_cross(depth_token_self, context=segment_token_self)
        segment_token_cross = self.fuse_transformer_segment_cross(segment_token_self, context=depth_token_self)

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

        def fuse_tokens(d_tok, s_tok, target_shape):
            d_p = self.fuse_proj_depth(d_tok)
            s_p = self.fuse_proj_segment(s_tok)
            gate = self.fuse_gate(torch.concat([d_p, s_p], dim=1))
            fused = gate * d_p + (1.0 - gate) * s_p
            return F.interpolate(fused, size=target_shape, mode="bilinear", align_corners=False)

        # --- Scales 5 down to 2 from Cross-Attention (for Gated Decoder) ---
        d5 = depth_token_cross[:, s5_idx : s5_idx + l5, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h5, w5)
        s5 = segment_token_cross[:, s5_idx : s5_idx + l5, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h5, w5)
        v5 = fuse_tokens(d5, s5, features["f5"].shape[-2:])

        d4 = depth_token_cross[:, s4_idx : s4_idx + l4, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h4, w4)
        s4 = segment_token_cross[:, s4_idx : s4_idx + l4, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h4, w4)
        v4 = fuse_tokens(d4, s4, features["f4"].shape[-2:])

        d3 = depth_token_cross[:, s3_idx : s3_idx + l3, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h3, w3)
        s3 = segment_token_cross[:, s3_idx : s3_idx + l3, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h3, w3)
        v3 = fuse_tokens(d3, s3, features["f3"].shape[-2:])

        d2 = depth_token_cross[:, s2_idx : s2_idx + l2, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h2, w2)
        s2 = segment_token_cross[:, s2_idx : s2_idx + l2, :].transpose(1, 2).contiguous().view(B, self.tok_dim, h2, w2)
        v2 = fuse_tokens(d2, s2, features["f2"].shape[-2:])

        # ---- Decode info from bottleneck up ----
        dec5 = self.dec5(torch.concat([v5, features["f5"]], dim=1))

        up_dec5 = F.interpolate(dec5, size=features["f4"].shape[-2:], mode="bilinear", align_corners=False)
        dec4 = self.dec4(torch.concat([v4, features["f4"], up_dec5], dim=1)) 

        up_dec4 = F.interpolate(dec4, size=features["f3"].shape[-2:], mode="bilinear", align_corners=False)
        dec3 = self.dec3(torch.concat([v3, features["f3"], up_dec4], dim=1)) 

        up_dec3 = F.interpolate(dec3, size=features["f2"].shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.concat([v2, features["f2"], up_dec3], dim=1)) 

        up_dec2 = F.interpolate(dec2, size=features["f1"].shape[-2:], mode="bilinear", align_corners=False)
        segment_logits = self.segment_head(vit_segment_self, up_dec2, features["f1"])
        depth_logits = self.depth_head(vit_depth_self, up_dec2, features["f1"])

        return depth_logits, segment_logits

if __name__ == "__main__":
    model = MultiHeadDecoder(tok_dim=128, num_heads=4, num_labels=40)
    im = torch.randn([1, 3, 288, 384])
    encoder = ResNetEncoder()
    features = encoder(im)
    model(features)
