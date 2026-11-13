"""Native PyTorch training and reusable epoch execution."""

import argparse
import csv
from functools import partial
import logging
import math
import os
from pathlib import Path
import platform

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from datasets import BAND_SPECS, CLASS_NAMES, MADOSDataset, segmentation_collate
from datasets.mados_dataset import read_split
from models import build_model
from utils.losses import build_loss
from utils.metrics import SegmentationMetrics, csv_metrics
from utils.seed import seed_everything, seed_worker
from utils.transforms import JointTransform
from utils.visualize import plot_training_curves


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data = config["dataset"]
    if not data.get("root"):
        raise ValueError(f"Set dataset.root in {path} to your MADOS directory")
    if data["input_channels"] != 11 or data["num_classes"] != 15:
        raise ValueError("The original MADOS reader requires 11 input bands and 15 classes")
    if data["band_names"] != [entry[0] for entry in BAND_SPECS]:
        raise ValueError("dataset.band_names must follow the documented MADOS band order")
    if data["class_names"] != list(CLASS_NAMES):
        raise ValueError("dataset.class_names must follow the original 15-class MADOS mapping")
    paths = ((data, "root"), (data, "cache_dir"), (config, "output_dir"), (config["evaluation"], "checkpoint"))
    for mapping, key in paths:
        value = mapping.get(key)
        if value is not None:
            value = Path(value).expanduser()
            if not value.is_absolute():
                value = path.parent / value
            mapping[key] = str(value.resolve())
    if (
        config["training"]["epochs"] < 1
        or config["training"]["batch_size"] < 1
        or config["evaluation"]["batch_size"] < 1
    ):
        raise ValueError("epochs and batch sizes must be positive")
    if config["training"]["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    if config["training"]["learning_rate"] <= 0 or config["training"]["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    # Check all official split lists before applying any debug sample limit.
    identifiers = {}
    for split in ("train", "val", "test"):
        split_path = Path(data["splits"][split])
        if not split_path.is_absolute():
            split_path = Path(data["root"]) / split_path
        identifiers[split] = set(read_split(split_path))
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = identifiers[left] & identifiers[right]
        if overlap:
            raise ValueError(f"Data leakage: {left}/{right} share IDs, including {sorted(overlap)[:3]}")
    return config


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Supported devices are CPU and CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    return device


def make_loader(config, split, device, training=False):
    data = config["dataset"]
    normalization = data["normalization"]
    transform = None
    if training:
        transform = JointTransform(
            config["augmentation"], normalization["mean"], data["ignore_index"], seed=config["training"]["seed"]
        )
    dataset = MADOSDataset(
        root=data["root"],
        split=split,
        split_file=data["splits"][split],
        mean=normalization["mean"],
        std=normalization["std"],
        ignore_index=data["ignore_index"],
        transform=transform,
        cache_dir=data["cache_dir"],
        max_samples=data["max_samples"].get(split),
    )
    generator = torch.Generator().manual_seed(config["training"]["seed"] + {"train": 0, "val": 1, "test": 2}[split])
    batch_size = config["training"]["batch_size"] if training else config["evaluation"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=bool(config["training"]["pin_memory"] and device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=partial(segmentation_collate, ignore_index=data["ignore_index"]),
    )
    return loader


def create_logger(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mados")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def run_epoch(
    model, loader, criterion, device, config, optimizer=None, description="Evaluation", prediction_writer=None
):
    training = optimizer is not None
    model.train(training)
    tracker = SegmentationMetrics(
        config["dataset"]["num_classes"], config["dataset"]["ignore_index"], config["metrics"]["absent_class_policy"]
    )
    numerator_sum = denominator_sum = 0.0
    annotated_pixels = empty_batches = 0
    progress = tqdm(loader, desc=description, leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        valid_pixels = int((targets != criterion.ignore_index).sum().item())
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            expected_shape = (images.shape[0], config["dataset"]["num_classes"], *targets.shape[1:])
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected_shape:
                raise ValueError(f"Model must return raw logits of shape {expected_shape}")
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite logits for samples {batch['id']}")
            numerator, denominator = criterion.components(logits, targets)
            loss = numerator / denominator.clamp_min(torch.finfo(logits.dtype).tiny)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for samples {batch['id']}")
            if training and valid_pixels > 0:
                loss.backward()
                clip_norm = config["training"].get("grad_clip_norm")
                if clip_norm is None:
                    clip_norm = float("inf")
                if clip_norm <= 0:
                    raise ValueError("grad_clip_norm must be positive or null")
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
                optimizer.step()
        # All-unlabelled batches contribute no statistics or optimizer update.
        # Counting them explicitly makes this visible in the CSV and log.
        empty_batches += int(valid_pixels == 0)
        annotated_pixels += valid_pixels
        numerator_sum += float(numerator.detach().item())
        denominator_sum += float(denominator.detach().item())
        predictions = logits.detach().argmax(dim=1)
        tracker.update(predictions, targets)
        if prediction_writer is not None:
            prediction_writer(batch, predictions)
        progress.set_postfix(loss=f"{numerator_sum / max(denominator_sum, 1e-30):.4f}", labelled=annotated_pixels)
    if denominator_sum == 0:
        raise ValueError(f"{description} has no annotated pixels. Check labels and crop settings.")
    result = tracker.compute()
    result.update(loss=numerator_sum / denominator_sum, annotated_pixels=annotated_pixels, empty_batches=empty_batches)
    return result


def build_optimizer(model, config):
    name = config["optimizer"].lower()
    arguments = {"lr": config["learning_rate"], "weight_decay": config["weight_decay"]}
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=config["momentum"], **arguments)
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {name}")
    return optimizers[name](model.parameters(), **arguments)


def build_scheduler(optimizer, config):
    options = config["scheduler"]
    name = options["name"].lower()
    if name == "none":
        return None
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=options["milestones"], gamma=options["gamma"])
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=options["min_learning_rate"]
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=options["factor"],
            patience=options["patience"],
            min_lr=options["min_learning_rate"],
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(config):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".training.lock"
    # Prevent two processes from writing the same run concurrently.
    with lock_path.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        csv_path = output / "metrics.csv"
        best_path = output / "checkpoints" / "best.pth"
        if csv_path.exists() or best_path.exists():
            raise FileExistsError(f"Existing experiment in {output}. Set output_dir to a new experiment directory.")
        for name in ("checkpoints", "logs", "curves", "predictions"):
            (output / name).mkdir(exist_ok=True)
        logger = create_logger(output / "logs" / "train.log")
        settings = config["training"]
        seed_everything(settings["seed"], settings["deterministic"])
        device = select_device(settings["device"])
        logger.info("Python %s, PyTorch %s, device=%s", platform.python_version(), torch.__version__, device)
        train_loader = make_loader(config, "train", device, training=True)
        val_loader = make_loader(config, "val", device)
        logger.info(
            "Train %d/%d, val %d/%d samples",
            len(train_loader.dataset),
            train_loader.dataset.official_split_size,
            len(val_loader.dataset),
            val_loader.dataset.official_split_size,
        )
        logger.info(
            "Class IDs 0..14, ignore_index=%d, macro absent-class policy=%s",
            config["dataset"]["ignore_index"],
            config["metrics"]["absent_class_policy"],
        )
        model = build_model(config["model"], config["dataset"]["input_channels"], config["dataset"]["num_classes"]).to(
            device
        )
        criterion = build_loss(config["loss"], config["dataset"]["num_classes"], config["dataset"]["ignore_index"]).to(
            device
        )
        optimizer = build_optimizer(model, settings)
        scheduler = build_scheduler(optimizer, settings)
        best_miou = -math.inf
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = None
            for epoch in range(1, settings["epochs"] + 1):
                learning_rate = optimizer.param_groups[0]["lr"]
                training_result = run_epoch(
                    model, train_loader, criterion, device, config, optimizer, f"Train {epoch}/{settings['epochs']}"
                )
                validation_result = run_epoch(
                    model, val_loader, criterion, device, config, description=f"Validate {epoch}/{settings['epochs']}"
                )
                row = {"epoch": epoch, "learning_rate": learning_rate}
                row.update(csv_metrics(training_result, "train"))
                row.update(csv_metrics(validation_result, "val"))
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
                if validation_result["miou"] > best_miou:
                    best_miou = validation_result["miou"]
                    save_checkpoint(
                        best_path,
                        {
                            "model_state_dict": model.state_dict(),
                            "epoch": epoch,
                            "best_miou": best_miou,
                            "config": config,
                            "validation": validation_result,
                            "torch_version": str(torch.__version__),
                        },
                    )
                    logger.info("Saved best checkpoint at epoch %d, val_mIoU=%.6f", epoch, best_miou)
                logger.info(
                    "Epoch %d | lr %.6g | train_loss %.6f | val_loss %.6f | val_mIoU %.6f | val_mF1 %.6f | empty_batches train=%d val=%d",
                    epoch,
                    learning_rate,
                    training_result["loss"],
                    validation_result["loss"],
                    validation_result["miou"],
                    validation_result["mf1"],
                    training_result["empty_batches"],
                    validation_result["empty_batches"],
                )
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(validation_result["loss"])
                elif scheduler is not None:
                    scheduler.step()
        plot_training_curves(csv_path, output / "curves")
        logger.info("Completed training. Best checkpoint: %s", best_path)
        return best_path
    finally:
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
