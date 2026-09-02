import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch import nn
from torch import cat
import torch
import torch.nn.functional as F
from models.encoder import ResNetEncoder

class SEGate(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid())

    def forward(self, x):
        squeeze = self.squeeze(x) # (B, C, 1, 1)
        squeeze = squeeze.flatten(1) # (B, C)

        gate = self.mlp(squeeze) # (B, C)
        gate = gate.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)

        return x * gate

class SegmentHead(nn.Module):
    def __init__(self, num_labels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.se_gate1 = SEGate(64)
        self.se_gate2 = SEGate(64)

        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dec4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_labels, kernel_size=3, padding=1))
                
    def forward(self, up3, features): # x is (B, 64, H, W)
        gated_features2 = self.se_gate2(features["f2"])
        up4 = self.dec3(cat([up3, gated_features2], dim=1))
        up4 = F.interpolate(up4, size=features["f1"].shape[2:], mode="bilinear", align_corners=False)

        gated_features1  = self.se_gate1(features["f1"])
        up5 = self.dec4(cat([up4, gated_features1], dim=1))
        # Upsample up5 to the original input resolution (2x of stem f1)
        up5 = self.up(up5)
        
        out = self.out(up5)
        return out

class DepthHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.se_gate1 = SEGate(64)
        self.se_gate2 = SEGate(64)

        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dec4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1))
        
    def forward(self, up3, features): # x is (B, 64, H, W)
        gated_features2 = self.se_gate2(features["f2"])
        up4 = self.dec3(cat([up3, gated_features2], dim=1))
        up4 = F.interpolate(up4, size=features["f1"].shape[2:], mode="bilinear", align_corners=False)

        gated_features1  = self.se_gate1(features["f1"])
        up5 = self.dec4(cat([up4, gated_features1], dim=1))
        # Upsample up5 to the original input resolution (2x of stem f1)
        up5 = self.up(up5)
        
        out = self.out(up5)
        return torch.clamp(out, min=1e-3, max=10.0)



class BoundaryHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.se_gate1 = SEGate(64)
        self.se_gate2 = SEGate(64)

        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dec4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1))
        
    def forward(self, up3, features): # x is (B, 64, H, W)
        gated_features2 = self.se_gate2(features["f2"])
        up4 = self.dec3(cat([up3, gated_features2], dim=1))
        up4 = F.interpolate(up4, size=features["f1"].shape[2:], mode="bilinear", align_corners=False)

        gated_features1  = self.se_gate1(features["f1"])
        up5 = self.dec4(cat([up4, gated_features1], dim=1))
        # Upsample up5 to the original input resolution (2x of stem f1)
        up5 = self.up(up5)
        
        out = self.out(up5)
        return out

class MultiHeadDecoder(nn.Module):
    def __init__(self, num_labels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.bot = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.depth_head = DepthHead()
        self.boundary_head = BoundaryHead()
        self.segment_head = SegmentHead(num_labels)

    def forward(self, encoded_features):
        bottleneck = encoded_features["f5"]

        up1 = self.bot(bottleneck)        
        up1 = F.interpolate(up1, size=encoded_features["f4"].shape[2:], mode="bilinear", align_corners=False)

        up2 = self.dec1(cat([up1, encoded_features["f4"]], dim=1))
        up2 = F.interpolate(up2, size=encoded_features["f3"].shape[2:], mode="bilinear", align_corners=False)

        up3 = self.dec2(cat([up2, encoded_features["f3"]], dim=1))
        up3 = F.interpolate(up3, size=encoded_features["f2"].shape[2:], mode="bilinear", align_corners=False)

        segment_logits = self.segment_head(up3, encoded_features)
        depth_logits = self.depth_head(up3, encoded_features)
        edge_logits = self.boundary_head(up3, encoded_features)

        return segment_logits, depth_logits, edge_logits


if __name__ == "__main__":
    encoder = ResNetEncoder()
    model = MultiHeadDecoder(num_labels=40)
    image = torch.randn((1, 3, 320, 640))

    features = encoder.forward(image)
    segment, depth, edge = model.forward(features)
    print(f"Output shape: Segment:{segment.shape}, Edge: {edge.shape}, Depth: {depth.shape}")
    segate = SEGate(3)
    out = segate.forward(image)
    print(out.shape)
