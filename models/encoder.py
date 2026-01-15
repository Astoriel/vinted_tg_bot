import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class ImageEncoder(nn.Module):
    """Frozen MobileNetV3-Large feature extractor.

    Input:  [B, 3, 224, 224]
    Output: [B, 960, 7, 7]
    """

    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.features(x)

    def train(self, mode=True):
        return super().train(False)
