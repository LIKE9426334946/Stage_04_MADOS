"""Read the original MADOS TIFF layout and official split files.

Band order, normalization statistics and label semantics follow the original
MADOS implementation by Ioannis Kakogeorgiou (MIT, 2023). See README.md.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
from torch.utils.data import Dataset


CLASS_NAMES = (
    "Marine Debris", "Dense Sargassum", "Sparse Floating Algae",
    "Natural Organic Material", "Ship", "Oil Spill", "Marine Water",
    "Sediment-Laden Water", "Foam", "Turbid Water", "Shallow Water",
    "Waves & Wakes", "Oil Platform", "Jellyfish", "Sea snot",
)

# name, filename wavelengths in nm, original pixel size in metres.
# Both wavelength variants were checked against the official ZIP directory.
BAND_SPECS = (
    ("B1", (442, 443), 60), ("B2", (492,), 10),
    ("B3", (559, 560), 10), ("B4", (665,), 10),
    ("B5", (704,), 20), ("B6", (739, 740), 20),
    ("B7", (780, 783), 20), ("B8", (833,), 10),
    ("B8A", (864, 865), 20), ("B11", (1610, 1614), 20),
    ("B12", (2186, 2202), 20),
)
BANDS_MEAN = (
    0.0582676, 0.05223386, 0.04381474, 0.0357083, 0.03412902,
    0.03680401, 0.03999107, 0.03566642, 0.03965081, 0.0267993, 0.01978944,
)
BANDS_STD = (
    0.03240627, 0.03432253, 0.0354812, 0.0375769, 0.03785412,
    0.04992323, 0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115,
)


@dataclass(frozen=True)
class Patch:
    sample_id: str
    image_paths: tuple[Path, ...]
    mask_path: Path


def read_split(path):
    """Read sample IDs from an official TXT split."""
    return Path(path).read_text(encoding="utf-8-sig").split()


class MADOSDataset(Dataset):
    """Return {'image': float32[C,H,W], 'mask': int64[H,W], 'id': str}.

    Raw labels 1..15 map to 0..14. Raw 0 maps to ignore_index. Unlabelled
    pixels are retained spatially and excluded by the loss and metrics.
    """

    def __init__(
        self, root, split="train", split_file=None, mean=BANDS_MEAN,
        std=BANDS_STD, ignore_index=-1, transform=None, cache_dir=None,
        max_samples=None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.ignore_index = int(ignore_index)
        self.split = split
        self.transform = transform
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.mean_tensor = torch.from_numpy(self.mean.copy())[:, None, None]
        self.std_tensor = torch.from_numpy(self.std.copy())[:, None, None]
        self.split_file = self.root / (split_file or f"splits/{split}_X.txt")
        identifiers = read_split(self.split_file)
        self.official_split_size = len(identifiers)
        identifiers = identifiers[:max_samples]
        self.samples = [self._index_patch(identifier) for identifier in identifiers]
        self.index_by_id = {sample.sample_id: i for i, sample in enumerate(self.samples)}
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _index_patch(self, sample_id):
        scene_name, crop = sample_id.rsplit("_", 1)
        scene_dir = self.root / scene_name
        mask_path = scene_dir / "10" / f"{scene_name}_L2R_cl_{crop}.tif"
        paths = list(scene_dir.glob(f"*/{scene_name}_L2R_rhorc_*_{crop}.tif"))
        paths.sort(key=lambda path: int(path.stem.split("_")[-2]))
        return Patch(sample_id, tuple(paths), mask_path)

    def __len__(self):
        return len(self.samples)

    def _cache_path(self, sample):
        signatures = []
        for path in (*sample.image_paths, sample.mask_path):
            stat = path.stat()
            signatures.append((str(path), stat.st_size, stat.st_mtime_ns))
        payload = json.dumps({"format": "mados-raw-nearest-v1", "sources": signatures})
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{sample.sample_id}_{key}.npz"

    @staticmethod
    def _read_tiffs(sample):
        with rasterio.open(sample.mask_path) as src:
            mask = src.read(1).astype(np.int64)
        bands = []
        for path in sample.image_paths:
            factor = int(path.parent.name) // 10
            with rasterio.open(path) as src:
                band = src.read(1, out_shape=(src.height * factor, src.width * factor), resampling=Resampling.nearest)
            bands.append(band)
        image = np.stack(bands).astype(np.float32)
        return image, mask

    def read_raw(self, index):
        """Read/cache the 11 raw reflectance bands and original 0..15 labels.

        The cache precedes augmentation, normalization and NaN imputation.
        Atomic replacement permits concurrent workers to build the same item.
        """
        sample = self.samples[index]
        if self.cache_dir is None:
            return self._read_tiffs(sample)
        cache_path = self._cache_path(sample)
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                return cached["image"].copy(), cached["mask"].copy()
        image, mask = self._read_tiffs(sample)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".tmp", delete=False) as handle:
                temporary_path = Path(handle.name)
                np.savez_compressed(handle, image=image, mask=mask)
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return image, mask

    def __getitem__(self, index):
        image, raw_mask = self.read_raw(index)
        # Preserve the author's imputation and normalization order.
        image = np.where(np.isnan(image), self.mean[:, None, None], image)
        mask = raw_mask - 1
        mask[raw_mask == 0] = self.ignore_index
        image = torch.from_numpy(np.ascontiguousarray(image))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        image = (image - self.mean_tensor) / self.std_tensor
        return {"image": image.contiguous(), "mask": mask.contiguous(), "id": self.samples[index].sample_id}


def segmentation_collate(samples, ignore_index=-1):
    """Pad variable-size samples with normalized zero / ignored labels."""
    channels = samples[0]["image"].shape[0]
    sizes = [tuple(sample["mask"].shape) for sample in samples]
    height = max(size[0] for size in sizes)
    width = max(size[1] for size in sizes)
    images = samples[0]["image"].new_zeros((len(samples), channels, height, width))
    masks = samples[0]["mask"].new_full((len(samples), height, width), ignore_index)
    for i, (sample, (h, w)) in enumerate(zip(samples, sizes)):
        images[i, :, :h, :w] = sample["image"]
        masks[i, :h, :w] = sample["mask"]
    return {"image": images, "mask": masks, "id": [s["id"] for s in samples], "sizes": sizes}
