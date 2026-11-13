"""Dataset-level multiclass metrics using segmentation_models_pytorch."""

import torch
from segmentation_models_pytorch import metrics as smp_metrics


class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index, absent_class_policy="exclude"):
        if absent_class_policy not in {"exclude", "include"}:
            raise ValueError("absent_class_policy must be 'exclude' or 'include'")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.absent_class_policy = absent_class_policy
        self.stats = torch.zeros((4, 1, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, predictions, targets):
        predictions = predictions.detach().to(device="cpu", dtype=torch.int64)
        targets = targets.detach().to(device="cpu", dtype=torch.int64)
        invalid = (targets != self.ignore_index) & ((targets < 0) | (targets >= self.num_classes))
        if invalid.any():
            raise ValueError("Metric targets contain invalid training labels")
        if (predictions < 0).any() or (predictions >= self.num_classes).any():
            raise ValueError("Metric predictions must be class IDs in 0..num_classes-1")
        batch_stats = smp_metrics.get_stats(
            predictions, targets, mode="multiclass", num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        for accumulated, current in zip(self.stats, batch_stats):
            accumulated.add_(current.sum(dim=0, keepdim=True))

    def compute(self):
        tp, fp, fn, tn = self.stats
        support = (tp + fn).squeeze(0)
        if support.sum() == 0:
            raise ValueError("This split/epoch contains no annotated pixels")
        union = (tp + fp + fn).squeeze(0)
        included = union > 0
        if self.absent_class_policy == "include":
            included = torch.ones_like(included)
        result = {"classes_in_average": int(included.sum()), "per_class": []}
        metric_functions = {
            "iou": smp_metrics.iou_score, "f1": smp_metrics.f1_score,
            "precision": smp_metrics.precision, "recall": smp_metrics.recall,
        }
        values = {}
        for name, function in metric_functions.items():
            score = function(tp, fp, fn, tn, reduction=None, zero_division=0.0).squeeze(0)
            values[name] = score
            aggregate_name = {"iou": "miou", "f1": "mf1"}.get(name, name)
            result[aggregate_name] = float(score[included].mean())
        for index in range(self.num_classes):
            row = {
                "class_id": index, "raw_label": index + 1,
                "support": int(support[index]),
                "predicted_pixels": int((tp + fp)[0, index]),
                "included_in_average": bool(included[index]),
            }
            for name, score in values.items():
                row[name] = float(score[index]) if included[index] else None
            result["per_class"].append(row)
        return result


def csv_metrics(result, prefix):
    row = {
        f"{prefix}_{key}": result[key]
        for key in ("loss", "miou", "mf1", "precision", "recall", "annotated_pixels", "empty_batches", "classes_in_average")
    }
    for values in result["per_class"]:
        index = values["class_id"]
        row[f"{prefix}_iou_{index:02d}"] = values["iou"]
        row[f"{prefix}_f1_{index:02d}"] = values["f1"]
    return row
