import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch import nn
from models.encoder import DepthAnythingEncoder
from models.multihead_decoder import MultiHeadDecoder
from torchinfo import summary
import torch.nn.functional as F

class TwinForge(nn.Module):
    def __init__(self, num_labels=41, num_heads=4, tok_dim=256, size=(378, 504), pretrained=True, freeze=True, freeze_early=False, max_frames=16):
        super().__init__()

        self.encoder = DepthAnythingEncoder(pretrained=pretrained, freeze=freeze, freeze_early=freeze_early)
        self.decoder = MultiHeadDecoder(num_labels, tok_dim, num_heads, size=size, max_frames=max_frames)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)
        B, T = x.shape[0], x.shape[1]

        features = self.encoder(x)
        refined_depth, segment_logits, depth_half, depth_quarter = self.decoder(features, B, T)

        # Only segmentation needs bilinear upsampling from 1/4 to full resolution
        segment_logits = F.interpolate(segment_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if refined_depth.shape[-2:] != x.shape[-2:]:
            pad_h = x.shape[-2] - refined_depth.shape[-2]
            pad_w = x.shape[-1] - refined_depth.shape[-1]
            if 0 <= pad_h <= 4 and 0 <= pad_w <= 4:
                refined_depth = F.pad(refined_depth, (0, pad_w, 0, pad_h), mode="replicate")
            else:
                refined_depth = F.interpolate(refined_depth, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.training:
            return segment_logits, refined_depth, depth_half, depth_quarter

        return segment_logits, refined_depth

if __name__ == "__main__":
    model = TwinForge(num_labels=41, num_heads=8, tok_dim=256, size=(378, 504), freeze=True)
    summary(model, input_size=(1, 4, 3, 378, 504))