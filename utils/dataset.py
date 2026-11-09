import os
import random
from glob import glob

import numpy as np
import torch

from torch.utils.data import Dataset

random.seed(23)
np.random.seed(23)
torch.manual_seed(23)


# Pixel-Level class distribution (total sum equals 1.0)
class_distr = torch.Tensor(
    [
        0.00336,
        0.00241,
        0.00336,
        0.00142,
        0.00775,
        0.18452,
        0.34775,
        0.20638,
        0.00062,
        0.1169,
        0.09188,
        0.01309,
        0.00917,
        0.00176,
        0.00963,
    ]
)

bands_mean = np.array(
    [
        0.0582676,
        0.05223386,
        0.04381474,
        0.0357083,
        0.03412902,
        0.03680401,
        0.03999107,
        0.03566642,
        0.03965081,
        0.0267993,
        0.01978944,
    ]
).astype("float32")

bands_std = np.array(
    [
        0.03240627,
        0.03432253,
        0.0354812,
        0.0375769,
        0.03785412,
        0.04992323,
        0.05884482,
        0.05545856,
        0.06423746,
        0.04211187,
        0.03019115,
    ]
).astype("float32")


def get_band(path):
    return int(path.split("_")[-2])  # 获取倒数第二个字符数字


class MADOS(Dataset):
    def __init__(self, path, splits, mode="train"):
        super().__init__()
        if mode == "train":
            self.ROIs_split = np.genfromtxt(os.path.join(splits, "train_X.txt"), dtype="str")
        elif mode == "test":
            self.ROIs_split = np.genfromtxt(os.path.join(splits, "test_X.txt"), dtype="str")
        elif mode == "val":
            self.ROIs_split = np.genfromtxt(os.path.join(splits, "val_X.txt"), dtype="str")
        else:
            raise
        self.X = []
        self.y = []

        self.tiles
