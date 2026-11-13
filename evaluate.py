"""Evaluate the best checkpoint and export multispectral segmentation results."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import torch

from models import build_model
from train import create_logger, load_config, make_loader, run_epoch, select_device
from utils.losses import build_loss
from utils.seed import seed_everything
from utils.visualize import PredictionWriter


def evaluate(config, checkpoint_path=None, split=None):
    output = Path(config["output_dir"])
    split = split or config["evaluation"]["split"]
    checkpoint_path = Path(checkpoint_path or config["evaluation"]["checkpoint"] or output / "checkpoints" / "best.pth")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown evaluation split: {split}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved_config = checkpoint["config"]
    for key in ("input_channels", "num_classes", "band_names", "class_names", "normalization", "ignore_index"):
        if saved_config["dataset"][key] != config["dataset"][key]:
            raise ValueError(f"dataset.{key} differs from the checkpoint's training configuration")
    if saved_config["model"] != config["model"]:
        raise ValueError("model configuration differs from the checkpoint's training configuration")
    seed_everything(config["training"]["seed"], config["training"]["deterministic"])
    device = select_device(config["training"]["device"])
    model = build_model(
        saved_config["model"], config["dataset"]["input_channels"], config["dataset"]["num_classes"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    criterion = build_loss(config["loss"], config["dataset"]["num_classes"], config["dataset"]["ignore_index"]).to(
        device
    )
    loader = make_loader(config, split, device)
    run_name = f"{split}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    result_dir = output / "predictions" / run_name
    result_dir.mkdir(parents=True, exist_ok=False)
    logger = create_logger(output / "logs" / "evaluate.log")
    logger.info(
        "Evaluating %d/%d %s samples using %s, checkpoint epoch %d",
        len(loader.dataset),
        loader.dataset.official_split_size,
        split,
        device,
        checkpoint["epoch"],
    )
    prediction_writer = None
    if config["evaluation"]["save_predictions"]:
        prediction_writer = PredictionWriter(
            loader.dataset, result_dir, config["evaluation"], config["dataset"]["class_names"]
        )
    result = run_epoch(
        model, loader, criterion, device, config, description=f"Evaluate {split}", prediction_writer=prediction_writer
    )
    for row, name in zip(result["per_class"], config["dataset"]["class_names"]):
        row["class_name"] = name
    report = {
        "split": split,
        "samples": len(loader.dataset),
        "official_split_samples": loader.dataset.official_split_size,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "macro_absent_class_policy": config["metrics"]["absent_class_policy"],
        "ignore_index": config["dataset"]["ignore_index"],
        "loss_config": config["loss"],
        "saved_samples": prediction_writer.saved if prediction_writer else [],
        **result,
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (result_dir / "per_class.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["per_class"][0]))
        writer.writeheader()
        writer.writerows(result["per_class"])
    logger.info(
        "Loss %.6f | mIoU %.6f | mF1 %.6f | macro precision %.6f | macro recall %.6f",
        result["loss"],
        result["miou"],
        result["mf1"],
        result["precision"],
        result["recall"],
    )
    logger.info("Results saved to %s", result_dir)
    return result_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"))
    args = parser.parse_args()
    evaluate(load_config(args.config), args.checkpoint, args.split)


if __name__ == "__main__":
    main()
