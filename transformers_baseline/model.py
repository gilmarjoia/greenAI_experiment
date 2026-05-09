"""
model.py — ViT-B/16 model wrapper for Fashion-MNIST classification.

Uses the pretrained ViT-B/16 from HuggingFace Transformers (google/vit-base-patch16-224).
The classification head is replaced with a new linear layer for 10 classes.

Architecture:
  - Backbone : ViT-B/16 (12 transformer layers, hidden_size=768, 12 attention heads)
  - Head     : Linear(768 → num_classes) — randomly initialised
  - Input    : (N, 3, 224, 224) — grayscale replicated to 3 channels and resized
"""

import torch
import torch.nn as nn
from transformers import ViTForImageClassification


class ViTBaseline(nn.Module):
    """
    Thin wrapper around HuggingFace's ViTForImageClassification.

    The pretrained backbone is loaded from 'google/vit-base-patch16-224'
    and its classifier head is replaced with a fresh Linear(768 → num_classes).
    All backbone weights are fine-tuned end-to-end.
    """

    MODEL_ID = "google/vit-base-patch16-224"

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Load pretrained ViT with a random head sized for num_classes
        self.vit = ViTForImageClassification.from_pretrained(
            self.MODEL_ID,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,  # replaces the 1000-class head
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: Pixel tensor of shape (N, 3, 224, 224)
        :return: Logits of shape (N, num_classes)
        """
        outputs = self.vit(pixel_values=x)
        return outputs.logits
