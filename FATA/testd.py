import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from easydict import EasyDict
from methods import tent

import models.Res as Resnet
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import os
from typing import Tuple, Callable, Any, Union

# import glob
from PIL import Image
import re
from einops import rearrange

# from dataset.selectedRotateImageFolder import prepare_test_data


class PairedINC(ImageFolder):
    def __init__(
        self,
        transform: Callable[..., Any] | None = None,
        corruption="gaussian_noise",
        level: Union[int, Tuple[int, ...]] = (1, 2, 3, 4, 5),
    ):
        super().__init__(
            os.path.join(
                "/ssd1/datasets/corruptions/ImageNet-C",
                corruption,
                str(level if type(level) is int else level[0]),
            ),
            transform,
        )

        self.imagenet_root = "/ssd1/datasets/ImageNet/val"
        self.level = level

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[index]
        corrupteds = []

        for lv in self.level:
            p = re.sub(r"/\d/", f"/{lv}/", path)
            imc = self.loader(p)
            if self.transform is not None:
                imc = self.transform(imc)
                corrupteds.append(imc)

        if self.target_transform is not None:
            target = self.target_transform(target)

        tokens = path.split("/")
        clsname = tokens[-2]
        filename, ext = os.path.splitext(tokens[-1])
        impath = os.path.join(self.imagenet_root, clsname, f"{filename}_{clsname}{ext}")
        im = self.loader(impath)
        if self.transform is not None:
            im = self.transform(im)
        return torch.stack([im] + corrupteds), target

    def __len__(self) -> int:
        return 5000
        # return super().__len__()
