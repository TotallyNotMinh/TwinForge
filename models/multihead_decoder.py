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

        self.pos_embed1 = nn.Parameter(torch.randn(1, 768, tok_dim) * 0.02)
        self.pos_embed2 = nn.Parameter(torch.randn(1, 432, tok_dim) * 0.02)
        self.pos_embed3 = nn.Parameter(torch.randn(1, 108, tok_dim) * 0.02)
        self.pos_embed4 = nn.Parameter(torch.randn(1, 48, tok_dim) * 0.02)
        self.pos_embed5 = nn.Parameter(torch.randn(1, 12, tok_dim) * 0.02)
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
            nn.Conv2d(2048 + 128, 512, kernel_size=1),
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
            nn.Conv2d(1024 * 2 + 128, 256, kernel_size=1),
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
            nn.Conv2d(512 * 2 + 128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(256 * 2 + 128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.segment_head = SegmentHead(tok_dim, num_labels)
        self.depth_head = DepthHead(tok_dim)

    def forward(self, features):
        B = features["f1"].shape[0]

        # Feature shapes:
        # torch.Size([1, 64, 144, 192])
        # torch.Size([1, 256, 72, 96])
        # torch.Size([1, 512, 36, 48])
        # torch.Size([1, 1024, 18, 24])
        # torch.Size([1, 2048, 9, 12])


        patch1 = self.patch_embedder1(features["f1"]) # torch.Size([1, 128, 24, 32])
        patch2 = self.patch_embedder2(features["f2"]) # torch.Size([1, 128, 18, 24])
        patch3 = self.patch_embedder3(features["f3"]) # torch.Size([1, 128, 9, 12])
        patch4 = self.patch_embedder4(features["f4"]) # torch.Size([1, 128, 6, 8])
        patch5 = self.patch_embedder5(features["f5"]) # torch.Size([1, 128, 3, 4])

        tok1 = self.flatten(patch1).transpose(2, 1) + self.pos_embed1 + self.level_embed.weight[0] # torch.Size([B, 768, 128])
        tok2 = self.flatten(patch2).transpose(2, 1) + self.pos_embed2 + self.level_embed.weight[1] # torch.Size([B, 432, 128])
        tok3 = self.flatten(patch3).transpose(2, 1) + self.pos_embed3 + self.level_embed.weight[2] # torch.Size([B, 108, 128])
        tok4 = self.flatten(patch4).transpose(2, 1) + self.pos_embed4 + self.level_embed.weight[3] # torch.Size([B, 48, 128])
        tok5 = self.flatten(patch5).transpose(2, 1) + self.pos_embed5 + self.level_embed.weight[4] # torch.Size([B, 12, 128])

        trans_output1 = self.transformer1(tok1) # torch.Size([B, 768, 128])
        trans_output2 = self.transformer2(tok2) # torch.Size([B, 432, 128])
        trans_output3 = self.transformer3(tok3) # torch.Size([B, 108, 128])
        trans_output4 = self.transformer4(tok4) # torch.Size([B, 48, 128])
        trans_output5 = self.transformer5(tok5) # torch.Size([B, 12, 128])

        
        joint_multi_layer_token = torch.concat([trans_output1, trans_output2, trans_output3, trans_output4, trans_output5], dim=-2)
        # Joint token shapes: torch.Size([B, 1368, 128])
        
        # Self Attention
        depth_token_self = self.fuse_transformer_depth_self(joint_multi_layer_token) # torch.Size([B, 1368, 128])
        segment_token_self = self.fuse_transformer_segment_self(joint_multi_layer_token) # torch.Size([B, 1368, 128])

        # Cross Attention
        depth_token_cross = self.fuse_transformer_depth_cross(depth_token_self, context=segment_token_self)
        segment_token_cross = self.fuse_transformer_segment_cross(segment_token_self, context=depth_token_self)

        # --- Scale 1 (Tokens 0:768 -> 24x32) from Self-Attention (for Task Heads) ---
        d1 = depth_token_self[:, 0:768, :].transpose(1, 2).contiguous().view(B, self.tok_dim, 24, 32)
        s1 = segment_token_self[:, 0:768, :].transpose(1, 2).contiguous().view(B, self.tok_dim, 24, 32)
        
        vit_depth_self = F.interpolate(d1, size=features["f1"].shape[-2:], mode="bilinear", align_corners=False)
        vit_segment_self = F.interpolate(s1, size=features["f1"].shape[-2:], mode="bilinear", align_corners=False)
        
        def fuse_tokens(d_tok, s_tok, target_shape):
            d_p = self.fuse_proj_depth(d_tok)
            s_p = self.fuse_proj_segment(s_tok)
            gate = self.fuse_gate(torch.concat([d_p, s_p], dim=1))
            fused = gate * d_p + (1.0 - gate) * s_p
            return F.interpolate(fused, size=target_shape, mode="bilinear", align_corners=False)

        # --- Scales 5 down to 2 from Cross-Attention (for Gated Decoder) ---
        # Scale 5 (Tokens 1356:1368 -> 3x4)
        d5 = depth_token_cross[:, 1356:1368, :].transpose(1, 2).contiguous().view(B, self.tok_dim, 3, 4)
        s5 = segment_token_cross[:, 1356:1368, :].transpose(1, 2).contiguous().view(B, self.tok_dim, 3, 4)
        v5 = fuse_tokens(d5, s5, features["f5"].shape[-2:])
        
        # Scale 4 (Tokens 1308:1356 -> 6x8)
        d4 = depth_token_cross[:, 1308:1356, :].transpose(1, 2).contiguous().view(B, self.tok_dim, 6, 8)
        s4 = segment_token_cross[:, 1308:1356, :].transpose(1, 2).contiguous().view(B, self.tok_dim, 6, 8)
        v4 = fuse_tokens(d4, s4, features["f4"].shape[-2:])

        # Scale 3 (Tokens 1200:1308 -> 9x12)
        d3 = depth_token_cross[:, 1200:1308, :].transpose(1, 2).contiguous().view(B, self.tok_dim, 9, 12)
        s3 = segment_token_cross[:, 1200:1308, :].transpose(1, 2).contiguous().view(B, self.tok_dim, 9, 12)
        v3 = fuse_tokens(d3, s3, features["f3"].shape[-2:])

        # Scale 2 (Tokens 768:1200 -> 18x24)
        d2 = depth_token_cross[:, 768:1200, :].transpose(1, 2).contiguous().view(B, self.tok_dim, 18, 24)
        s2 = segment_token_cross[:, 768:1200, :].transpose(1, 2).contiguous().view(B, self.tok_dim, 18, 24)
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
