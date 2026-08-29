from torch import nn
from .encoder import ResNetEncoder
from .multihead_decoder import MultiHeadDecoder

class TwinForge(nn.Module):
    def __init__(self, num_labels, pretrained=True, freeze=True):
        super().__init__()

        self.encoder = ResNetEncoder(pretrained=pretrained, freeze=freeze)
        self.decoder = MultiHeadDecoder(num_labels)

    def forward(self, x):
        features = self.encoder(x)
        segment_logits, depth_logits, edge_logits = self.decoder(features)
        return segment_logits, depth_logits, edge_logits