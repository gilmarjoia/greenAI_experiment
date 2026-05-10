"""
model.py — Transformer model wrapper for Fashion-MNIST classification.

Uses a lightweight pretrained Transformer from HuggingFace (facebook/deit-tiny-patch16-224).
The classification head is replaced with a new linear layer for 10 classes.

Architecture:
  - Backbone : DeiT-Tiny (12 layers, hidden_size=192, 3 attention heads)
  - Head     : Linear(192 → num_classes) — randomly initialised
  - Input    : (N, 3, 224, 224) — grayscale replicated to 3 channels and resized
"""
import os

import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

login(os.getenv("HF_TOKEN"))


class TransformerBaseline(nn.Module):
    """
    Thin wrapper around HuggingFace's AutoModelForImageClassification.

    The pretrained backbone is loaded from 'facebook/deit-tiny-patch16-224'
    and its classifier head is replaced with a fresh Linear(192 → num_classes).
    All backbone weights are fine-tuned end-to-end.
    """

    MODEL_ID = "facebook/deit-tiny-patch16-224"

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Load pretrained model with a random head sized for num_classes
        self.model = AutoModelForImageClassification.from_pretrained(
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
        outputs = self.model(pixel_values=x)
        return outputs.logits
