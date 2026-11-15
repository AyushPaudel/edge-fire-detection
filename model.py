"""
Lightweight fire/not_fire classifier for Jetson Nano, with built-in support
for Class Activation Mapping (CAM) — a rough "where is the fire" heatmap
computed from a single forward pass (no backprop needed), so it stays cheap
enough to run on the Nano via TensorRT.

Architecture requirement for CAM to work: Global Average Pool directly into
a single Linear layer (no extra hidden FC layers in between). The heatmap is
then just a weighted sum of the last conv feature maps, using that Linear
layer's weights.
"""
import torch
import torch.nn as nn
import torchvision


class FireClassifier(nn.Module):
    """
    Binary fire classifier (single logit, sigmoid -> P(fire)).

    forward(x, return_features=True) also returns the last conv feature map
    (before pooling) so a CAM heatmap can be computed.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_small(
            weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        self.features = backbone.features  # -> (B, 576, H/32, W/32)
        self.feat_channels = 576
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.feat_channels, 1)  # single logit: P(fire)

    def forward(self, x, return_features: bool = False):
        feat = self.features(x)              # (B, 576, h, w)
        pooled = self.gap(feat).flatten(1)    # (B, 576)
        logit = self.fc(pooled)               # (B, 1)
        if return_features:
            return logit, feat
        return logit

    def cam_weights(self):
        """The fc weight vector used to build a CAM heatmap. Shape: (576,)"""
        return self.fc.weight.detach().cpu().numpy().reshape(-1)


if __name__ == "__main__":
    m = FireClassifier(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logit, feat = m(x, return_features=True)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"logit shape: {logit.shape}, feature map shape: {feat.shape}")
    print(f"Params: {n_params/1e6:.2f}M")