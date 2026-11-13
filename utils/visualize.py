"""Training curves and multispectral segmentation exports."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
import numpy as np
from PIL import Image
import rasterio


# Original MADOS colour names, in raw label order 1..15.
CLASS_COLORS = (
    "red", "green", "limegreen", "brown", "orange", "thistle", "navy",
    "gold", "purple", "darkkhaki", "darkturquoise", "bisque", "dimgrey",
    "hotpink", "yellow",
)


def plot_training_curves(csv_path, output_dir):
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in rows]
    for filename, metric, title in (("loss.png", "loss", "Loss"), ("miou.png", "miou", "Mean IoU")):
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for split in ("train", "val"):
            ax.plot(epochs, [float(row[f"{split}_{metric}"]) for row in rows], label=split, marker=".")
        ax.set(xlabel="Epoch", ylabel=title, title=title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, (("miou", "Mean IoU"), ("mf1", "Mean F1"), ("precision", "Macro precision"), ("recall", "Macro recall"))):
        for split in ("train", "val"):
            ax.plot(epochs, [float(row[f"{split}_{metric}"]) for row in rows], label=split, marker=".")
        ax.set(xlabel="Epoch", ylabel=title, title=title, ylim=(0, 1))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output_dir / "metrics.png", dpi=160)
    plt.close(fig)


def make_rgb(image, indices=(3, 2, 1), percentiles=(2, 98)):
    rgb = np.zeros((*image.shape[1:], 3), dtype=np.float32)
    for channel, band_index in enumerate(indices):
        band = image[band_index]
        valid = np.isfinite(band)
        if valid.any():
            low, high = np.percentile(band[valid], percentiles)
            if high > low:
                rgb[:, :, channel][valid] = np.clip((band[valid] - low) / (high - low), 0, 1)
    return np.round(rgb * 255).astype(np.uint8)


def colorize(raw_labels):
    palette = np.zeros((16, 3), dtype=np.uint8)
    palette[1:] = np.round(np.array([to_rgb(color) for color in CLASS_COLORS]) * 255).astype(np.uint8)
    return palette[raw_labels]


class PredictionWriter:
    def __init__(self, dataset, output_dir, config, class_names):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.config = config
        self.class_names = class_names
        self.saved = []
        self.max_saved = config.get("max_saved_predictions")

    def __call__(self, batch, predictions):
        predictions = predictions.detach().cpu().numpy()
        for sample_id, shape, prediction in zip(batch["id"], batch["sizes"], predictions):
            if self.max_saved is not None and len(self.saved) >= self.max_saved:
                return
            index = self.dataset.index_by_id[sample_id]
            image, target = self.dataset.read_raw(index)
            prediction = prediction[:shape[0], :shape[1]].astype(np.uint8) + 1
            self._save_sample(index, image, target.astype(np.uint8), prediction)

    def _save_sample(self, index, image, target, prediction):
        sample = self.dataset.samples[index]
        folder = self.output_dir / sample.sample_id
        folder.mkdir(parents=True, exist_ok=True)
        # The original TIFFs may have CRS metadata but no real affine georeference.
        with rasterio.open(sample.image_paths[1]) as reference:
            transform = reference.transform
            crs = reference.crs
            geo_available = not transform.is_identity and crs is not None
        profile = {
            "driver": "GTiff", "height": target.shape[0], "width": target.shape[1],
            "compress": "deflate", "crs": crs,
        }
        if not transform.is_identity:
            profile["transform"] = transform
        arrays = (("image.tif", image.astype(np.float32), None),
                  ("target.tif", target[None], 0),
                  ("prediction.tif", prediction[None], None))
        for name, array, nodata in arrays:
            with rasterio.open(folder / name, "w", count=array.shape[0], dtype=array.dtype, nodata=nodata, **profile) as destination:
                destination.write(array)
                destination.update_tags(sample_id=sample.sample_id, georeferenced=str(geo_available).lower())
                if name == "image.tif":
                    for number, band_path in enumerate(sample.image_paths, 1):
                        destination.set_band_description(number, band_path.stem.split("_")[-2] + " nm")
                else:
                    destination.update_tags(label_encoding="raw MADOS: 0 unlabelled, classes 1..15")
        rgb = make_rgb(image, self.config["rgb_indices"], self.config["rgb_percentiles"])
        target_color, prediction_color = colorize(target), colorize(prediction)
        Image.fromarray(rgb).save(folder / "rgb.png")
        Image.fromarray(target_color).save(folder / "target.png")
        Image.fromarray(prediction_color).save(folder / "prediction.png")
        overlay = np.round(0.55 * rgb.astype(float) + 0.45 * prediction_color.astype(float)).astype(np.uint8)
        fig, axes = plt.subplots(1, 4, figsize=(14, 5.2))
        for ax, content, title in zip(axes, (rgb, target_color, prediction_color, overlay), ("RGB preview", "Ground truth", "Prediction", "Overlay")):
            ax.imshow(content)
            ax.set_title(title)
            ax.axis("off")
        handles = [Patch(facecolor="black", label="Unlabelled")]
        handles += [Patch(facecolor=color, label=name) for color, name in zip(CLASS_COLORS, self.class_names)]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(sample.sample_id)
        fig.subplots_adjust(left=0.015, right=0.985, top=0.87, bottom=0.27, wspace=0.08)
        fig.savefig(folder / "comparison.png", dpi=140)
        plt.close(fig)
        self.saved.append({"sample_id": sample.sample_id, "georeferenced": geo_available})
