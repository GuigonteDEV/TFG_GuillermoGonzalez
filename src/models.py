
import torch
import torch.nn as nn
from torchvision import models



class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = models.efficientnet_b3(weights="IMAGENET1K_V1")

        # Freeze backbone completo
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_last_fc(self):
        for param in self.backbone.features[-1].parameters():
                param.requires_grad = True
                

class MLPBinary(nn.Module):
    """
    Cabeza binaria: Detecta si el parche es Normal (0) o Patológico (1).
    """
    def __init__(self, input_dim=1024, num_classes=6):
        super(MLPBinary, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
    

class MLP(nn.Module):
    """
    Clasificador multiclase para nivel 2 jerárquico.
    """
    def __init__(self, input_dim=1024, num_classes=6):  # ← default correcto
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)