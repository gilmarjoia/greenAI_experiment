"""
model.py — Simple CNN baseline for Fashion-MNIST classification.

Architecture mirrors the complexity tier of YOLO26n-cls (nano classification):
3 Conv blocks + 2 FC layers, no pretrained weights.
"""
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    A simple CNN for 28×28 image classification (10 classes).

    Input expected: (N, 3, 28, 28)  — grayscale replicated to 3 channels
    to match YOLO's preprocessing pipeline.
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        # Block 1 — 3×28×28 → 32×28×28 → 32×14×14
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 → 14
        )

        # Block 2 — 32×14×14 → 64×14×14 → 64×7×7
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14 → 7
        )

        # Block 3 — 64×7×7 → 128×7×7 (no pooling, keep spatial info)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.
        :param x:
        :return:
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x