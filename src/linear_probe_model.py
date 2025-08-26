import torch
import torch.nn as nn


class CLIPWithLinearProbe(nn.Module):
    def __init__(self, clip_model, num_classes, dropout=0.5):
        super().__init__()
        self.clip = clip_model

        # Freeze backbone
        for param in self.clip.parameters():
            param.requires_grad = False

        # Get feature dimension
        dummy = torch.randn(1, 3, 224, 224)  # adjust input size if needed
        feature_dim = self.clip.visual(dummy).shape[1]
        print(f"Visual feature dimension: {feature_dim}")

        # Linear head (trainable)
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(feature_dim, num_classes))
        self.linear_head = nn.Sequential(*layers)  # keep it separate

    def forward(self, image=None, text=None):
        # Get frozen features from visual backbone
        features = self.clip.visual(image)  # [B, feature_dim]
        # Pass through trainable linear head
        logits = self.linear_head(features)
        return logits

