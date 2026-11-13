"""Pixel-additive losses with explicit treatment of unlabelled pixels."""

import torch
from torch import nn
import torch.nn.functional as F


# Published class distribution, in the original 15-class order.
CLASS_DISTRIBUTION = (
    0.00336, 0.00241, 0.00336, 0.00142, 0.00775, 0.18452,
    0.34775, 0.20638, 0.00062, 0.1169, 0.09188, 0.01309,
    0.00917, 0.00176, 0.00963,
)


class SegmentationLoss(nn.Module):
    def __init__(self, config, num_classes, ignore_index):
        super().__init__()
        self.name = config["name"]
        self.ignore_index = ignore_index
        self.label_smoothing = float(config.get("label_smoothing", 0.0))
        self.gamma = float(config.get("focal_gamma", 2.0))
        weights = config.get("class_weights")
        if isinstance(weights, str):
            c = float(config.get("weight_parameter", 1.03))
            weights = 1 / torch.log(c + torch.tensor(CLASS_DISTRIBUTION, dtype=torch.float32))
        elif weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", weights)

    def components(self, logits, targets):
        """Return a differentiable numerator and its averaging denominator.

        Accumulating these quantities gives a valid pixel-weighted epoch loss,
        even when batches have very different amounts of annotation.
        """
        valid = targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0, logits.new_zeros(())
        pixel_loss = F.cross_entropy(
            logits, targets, weight=self.weights, ignore_index=self.ignore_index,
            reduction="none", label_smoothing=self.label_smoothing,
        )
        if self.name == "focal":
            safe_targets = targets.masked_fill(~valid, 0)
            correct_probability = logits.softmax(dim=1).gather(1, safe_targets[:, None]).squeeze(1)
            pixel_loss = pixel_loss * (1 - correct_probability).pow(self.gamma)
        denominator = valid.sum().to(logits.dtype)
        if self.weights is not None:
            denominator = self.weights[targets[valid]].sum()
        return pixel_loss[valid].sum(), denominator

    def forward(self, logits, targets):
        numerator, denominator = self.components(logits, targets)
        return numerator / denominator.clamp_min(torch.finfo(logits.dtype).tiny)


def build_loss(config, num_classes, ignore_index):
    return SegmentationLoss(config, num_classes, ignore_index)
