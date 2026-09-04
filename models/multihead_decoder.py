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

        self.transformer1 = TransformerBlock(tok_dim, num_heads)
        self.transformer2 = TransformerBlock(tok_dim, num_heads)
        self.transformer3 = TransformerBlock(tok_dim, num_heads)
        self.transformer4 = TransformerBlock(tok_dim, num_heads)
        self.transformer5 = TransformerBlock(tok_dim, num_heads)

        self.fuse_transformer_depth = TransformerBlock(tok_dim, num_heads)
        self.fuse_transformer_segment = TransformerBlock(tok_dim, num_heads)

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


        patch1 = self.patch_embedder1(features["f1"]) 
        patch2 = self.patch_embedder2(features["f2"])
        patch3 = self.patch_embedder3(features["f3"])
        patch4 = self.patch_embedder4(features["f4"])
        patch5 = self.patch_embedder5(features["f5"])

        # Patch shapes
        # torch.Size([1, 128, 24, 32])
        # torch.Size([1, 128, 18, 24])
        # torch.Size([1, 128, 9, 12])
        # torch.Size([1, 128, 6, 8])
        # torch.Size([1, 128, 3, 4])


        tok1 = self.flatten(patch1).transpose(2, 1)
        tok2 = self.flatten(patch2).transpose(2, 1)
        tok3 = self.flatten(patch3).transpose(2, 1)
        tok4 = self.flatten(patch4).transpose(2, 1)
        tok5 = self.flatten(patch5).transpose(2, 1)

        # Token shapes
        # torch.Size([1, 768, 128])
        # torch.Size([1, 432, 128])
        # torch.Size([1, 108, 128])
        # torch.Size([1, 48, 128])
        # torch.Size([1, 12, 128])


        trans_output1 = self.transformer1(tok1)
        trans_output2 = self.transformer2(tok2)
        trans_output3 = self.transformer3(tok3)
        trans_output4 = self.transformer4(tok4)
        trans_output5 = self.transformer5(tok5)

        # Transformer output shapes
        # torch.Size([1, 768, 128])
        # torch.Size([1, 432, 128])
        # torch.Size([1, 108, 128])
        # torch.Size([1, 48, 128])
        # torch.Size([1, 12, 128])


        joint_multi_layer_token = torch.concat([trans_output1, trans_output2, trans_output3, trans_output4, trans_output5], dim=-2)
        # Joint token shapes: torch.Size([1, 1368, 128])
        

        depth_token = self.fuse_transformer_depth(joint_multi_layer_token)
        segment_token = self.fuse_transformer_segment(joint_multi_layer_token)
        # Depth/Segment token shapes: torch.Size([1, 1368, 128])

        vit_fe1 = trans_output1.transpose(1, 2).contiguous().view(B, self.tok_dim, patch1.shape[-2], patch1.shape[-1])  # (B, tok_dim, 24, 32)
        vit_fe2 = trans_output2.transpose(1, 2).contiguous().view(B, self.tok_dim, patch2.shape[-2], patch2.shape[-1])  # (B, tok_dim, 18, 24)
        vit_fe3 = trans_output3.transpose(1, 2).contiguous().view(B, self.tok_dim, patch3.shape[-2], patch3.shape[-1])  # (B, tok_dim, 9, 12)
        vit_fe4 = trans_output4.transpose(1, 2).contiguous().view(B, self.tok_dim, patch4.shape[-2], patch4.shape[-1])  # (B, tok_dim, 6, 8)
        vit_fe5 = trans_output5.transpose(1, 2).contiguous().view(B, self.tok_dim, patch5.shape[-2], patch5.shape[-1])  # (B, tok_dim, 3, 4)

        # Scale each vit_fe back to its layer's spatial resolution:
        v5 = F.interpolate(vit_fe5, size=features["f5"].shape[-2:], mode="bilinear", align_corners=False) # (B, tok_dim, 9, 12)
        v4 = F.interpolate(vit_fe4, size=features["f4"].shape[-2:], mode="bilinear", align_corners=False) # (B, tok_dim, 18, 24)
        v3 = F.interpolate(vit_fe3, size=features["f3"].shape[-2:], mode="bilinear", align_corners=False) # (B, tok_dim, 36, 48)
        v2 = F.interpolate(vit_fe2, size=features["f2"].shape[-2:], mode="bilinear", align_corners=False) # (B, tok_dim, 72, 96)
        v1 = F.interpolate(vit_fe1, size=features["f1"].shape[-2:], mode="bilinear", align_corners=False) # (B, tok_dim, 72, 96)


        dec5 = self.dec5(torch.concat([v5, features["f5"]], dim=1))

        up_dec5 = F.interpolate(dec5, size=features["f4"].shape[-2:], mode="bilinear", align_corners=False)
        dec4 = self.dec4(torch.concat([v4, features["f4"], up_dec5], dim=1)) 

        up_dec4 = F.interpolate(dec4, size=features["f3"].shape[-2:], mode="bilinear", align_corners=False)
        dec3 = self.dec3(torch.concat([v3, features["f3"], up_dec4], dim=1)) 

        up_dec3 = F.interpolate(dec3, size=features["f2"].shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.concat([v2, features["f2"], up_dec3], dim=1)) 

        up_dec2 = F.interpolate(dec2, size=features["f1"].shape[-2:], mode="bilinear", align_corners=False)
        segment_logits = self.segment_head(v1, up_dec2, features["f1"])
        depth_logits = self.depth_head(v1, up_dec2, features["f1"])

        return depth_logits, segment_logits

if __name__ == "__main__":
    model = MultiHeadDecoder(tok_dim=128, num_heads=4, num_labels=40)
    im = torch.randn([1, 3, 288, 384])
    print(model(im))