"""
model.py — Transformer model wrapper for transformers_modified.

Identical to transformers_baseline/model.py.
The same DeiT-Tiny backbone is used; improvements come from better
training hyperparameters and stronger data augmentation.
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
