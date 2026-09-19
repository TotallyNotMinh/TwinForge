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
    def __init__(self, num_labels=41, num_heads=4, tok_dim=256, size=(384, 512), pretrained=True, freeze=True, freeze_early=False, max_frames=16):
        super().__init__()

        self.encoder = DepthAnythingEncoder(pretrained=pretrained, freeze=freeze, freeze_early=freeze_early)
        self.decoder = MultiHeadDecoder(num_labels, tok_dim, num_heads, size=size, max_frames=max_frames)

    def forward(self, x):
        B, T = x.shape[0], x.shape[1]

        features = self.encoder(x)
        depth_logits, segment_logits = self.decoder(features, B, T)

        # Upsample by 2x back to input image resolution 
        segment_logits = F.interpolate(segment_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        depth_logits = F.interpolate(depth_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return segment_logits, depth_logits

if __name__ == "__main__":
    model = TwinForge(num_labels=41, num_heads=8, tok_dim=256, size=(384, 512), freeze=True)
    summary(model, input_size=(1, 8, 3, 384, 512))