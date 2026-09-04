import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch import nn
from models.encoder import ResNetEncoder
from models.multihead_decoder import MultiHeadDecoder
from torchinfo import summary

class TwinForge(nn.Module):
    def __init__(self, num_labels, num_heads, tok_dim , pretrained=True, freeze=False):
        super().__init__()

        self.encoder = ResNetEncoder(pretrained=pretrained, freeze=freeze)
        self.decoder = MultiHeadDecoder(num_labels, tok_dim, num_heads)

    def forward(self, x):
        features = self.encoder(x)
        depth_logits, segment_logits = self.decoder(features)
        return segment_logits, depth_logits

if __name__ == "__main__":
    model = TwinForge(num_labels=41, num_heads=8, tok_dim=128, freeze=False)
    summary(model, input_size=(1, 3, 288, 384))