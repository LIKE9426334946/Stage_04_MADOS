"""Joint spatial augmentation for all reflectance bands and sparse labels."""

import torch
import torch.nn.functional as F
from torch.utils.data import get_worker_info


class JointTransform:
    def __init__(self, config, mean, ignore_index, seed=0):
        self.scale_range = tuple(config.get("scale_range", [1.0, 1.0]))
        self.crop_size = config.get("crop_size")
        self.flip_probabilities = tuple(config.get("flip_probabilities", [0.2, 0.2, 0.2, 0.4]))
        self.rotation_probability = float(config.get("rotation_probability", 0.8))
        self.ensure_labeled_crop = bool(config.get("ensure_labeled_crop", True))
        self.crop_attempts = int(config.get("crop_attempts", 10))
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.ignore_index = ignore_index
        self.seed = int(seed)
        self.generator = None
        self.generator_seed = None
        if len(self.scale_range) != 2 or not 0 < self.scale_range[0] <= self.scale_range[1]:
            raise ValueError("scale_range must be [positive_minimum, maximum]")
        if self.crop_size is not None and (len(self.crop_size) != 2 or any(int(v) != v or v <= 0 for v in self.crop_size)):
            raise ValueError("crop_size must be [positive_height, positive_width] or null")
        if len(self.flip_probabilities) != 4 or any(p < 0 for p in self.flip_probabilities) or abs(sum(self.flip_probabilities) - 1) > 1e-6:
            raise ValueError("flip_probabilities must contain horizontal, vertical, both, none probabilities summing to 1")
        if not 0 <= self.rotation_probability <= 1 or self.crop_attempts < 1:
            raise ValueError("Invalid rotation_probability or crop_attempts")

    def __call__(self, image, mask):
        # Keep augmentation independent of model initialization and dropout.
        # DataLoader assigns a distinct, reproducible seed to each worker.
        worker = get_worker_info()
        seed = self.seed if worker is None else worker.seed
        if self.generator is None or self.generator_seed != seed:
            self.generator = torch.Generator().manual_seed(seed)
            self.generator_seed = seed
        generator = self.generator
        low, high = self.scale_range
        scale = low + (high - low) * torch.rand((), generator=generator).item()
        if scale != 1.0:
            size = [max(1, round(side * scale)) for side in mask.shape]
            image = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]
            mask = F.interpolate(mask[None, None].float(), size=size, mode="nearest")[0, 0].long()
        flip = int(torch.multinomial(torch.tensor(self.flip_probabilities), 1, generator=generator).item())
        dimensions = ((-1,), (-2,), (-2, -1), ())
        if dimensions[flip]:
            image = torch.flip(image, dimensions[flip])
            mask = torch.flip(mask, dimensions[flip])
        if torch.rand((), generator=generator).item() < self.rotation_probability:
            k = int(torch.randint(0, 4, (), generator=generator).item())
            image = torch.rot90(image, k, (-2, -1))
            mask = torch.rot90(mask, k, (-2, -1))
        if self.crop_size is not None:
            image, mask = self._crop(image, mask, generator)
        return image.contiguous(), mask.contiguous()

    def _crop(self, image, mask, generator):
        crop_h, crop_w = map(int, self.crop_size)
        h, w = mask.shape
        padded_h, padded_w = max(h, crop_h), max(w, crop_w)
        if (padded_h, padded_w) != (h, w):
            # Mean reflectance becomes zero after normalization.
            padded_image = self.mean.expand(-1, padded_h, padded_w).clone()
            padded_mask = mask.new_full((padded_h, padded_w), self.ignore_index)
            top, left = (padded_h - h) // 2, (padded_w - w) // 2
            padded_image[:, top:top + h, left:left + w] = image
            padded_mask[top:top + h, left:left + w] = mask
            image, mask = padded_image, padded_mask
        h, w = mask.shape
        for _ in range(self.crop_attempts):
            top = int(torch.randint(0, h - crop_h + 1, (), generator=generator).item())
            left = int(torch.randint(0, w - crop_w + 1, (), generator=generator).item())
            candidate = mask[top:top + crop_h, left:left + crop_w]
            if not self.ensure_labeled_crop or (candidate != self.ignore_index).any():
                return image[:, top:top + crop_h, left:left + crop_w], candidate
        valid = torch.nonzero(mask != self.ignore_index)
        if self.ensure_labeled_crop and len(valid):
            y, x = valid[int(torch.randint(len(valid), (), generator=generator).item())].tolist()
            top = int(torch.randint(max(0, y - crop_h + 1), min(y, h - crop_h) + 1, (), generator=generator).item())
            left = int(torch.randint(max(0, x - crop_w + 1), min(x, w - crop_w) + 1, (), generator=generator).item())
        return image[:, top:top + crop_h, left:left + crop_w], mask[top:top + crop_h, left:left + crop_w]
