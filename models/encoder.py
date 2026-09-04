import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )

        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = {}

        x = self.stem(x)
        features["f1"] = x # 64 channels

        x = self.maxpool(x)
        x = self.layer1(x)
        features["f2"] = x # 64 channels

        x = self.layer2(x)
        features["f3"] = x # 128 channels

        x = self.layer3(x)
        features["f4"] = x # 256 channels

        x = self.layer4(x)
        features["f5"] = x # 512 channels

        return features


