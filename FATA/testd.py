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


def layer_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
    module._feature = output.detach().cpu()
    module._style_mean = (
        output.mean((-1, -2)).detach().cpu()
    )  # style: [B C] <- output: [B C H W]


def extract_fts(net, loader, prefix="imagenet_val"):
    # target_layers1 = [net1.layer1, net1.layer2, net1.layer3, net1.layer4]/
    target_layers = [net.layer1, net.layer2, net.layer3, net.layer4]

    # Y = []
    # fts = [[] for _ in range(len(target_layers))]   #
    # styles = [[] for _ in range(len(target_layers))]   #
    # embds = []
    # logits = []
    # correct = []
    for i, (x, y) in enumerate(tqdm(loader)):
        print(x.size())
        B, N = x.size()[:2]
        x = x.cuda()  # B N C H W
        x = rearrange(x, "B N C H W -> (B N) C H W")
        x, e = net(x, return_feature=True)  # x: BN L, ft: BN C

        fts = []
        styles = []
        for j, layer in enumerate(target_layers):
            f = rearrange(layer._feature, "(B N) C H W -> N B C H W", B=B, N=N)
            s = rearrange(layer._style_mean, "(B N) C -> N B C", B=B, N=N)
            fts.append(f.detach().cpu())
            styles.append(s.detach().cpu())

        yy = y.repeat(N)
        c = x.argmax(dim=1).detach().cpu() == yy  # (BN)   x: BN C H W,
        c = rearrange(c, "(B N) -> N B", B=B, N=N)
        # correct.append(c)

        x = rearrange(x, "(B N) C -> N B C", B=B, N=N)
        e = rearrange(e, "(B N) C -> N B C", B=B, N=N)
        # embds.append(e.detach().cpu())
        # logits.append(x.detach().cpu())
        # Y.append(y)

        torch.save(
            {
                "features": fts,
                "embeddings": e.detach().cpu(),
                # 'ifeatures': [torch.concat(f) for f in ifts],
                "labels": y.cpu(),
                "logits": x.detach().cpu(),
                "styles": styles,
                # 'ilogits': torch.concat(ilogits),
                "correct": c.detach().cpu(),
            },
            f"/ssd1/tta/{prefix}_resnet50_bn_INC0-5_{i:02d}.pth",
        )

        # ImageNet
        # im, ift = net1(im, return_feature=True)
        # for j, layer in enumerate(target_layers1):
        #     ifts[j].append(layer._style_mean.cpu())
        # ifts[-1].append(ift.detach().cpu())
        # ilogits.append(im.detach().cpu())

        # #IN-C
        # x, ft = net2(x, return_feature=True)
        # x: torch.Tensor
        # for j, layer in enumerate(target_layers2):
        #     fts[j].append(layer._style_mean.cpu())
        # fts[-1].append(ft.detach().cpu())
        # logits.append(x.detach().cpu())

    torch.save(
        {
            "features": [torch.concat(f, 1) for f in fts],
            "embeddings": torch.concat(embds, 1),
            # 'ifeatures': [torch.concat(f) for f in ifts],
            "labels": torch.concat(Y),
            "logits": torch.concat(logits, 1),
            "styles": [torch.concat(s, 1) for s in styles],
            # 'ilogits': torch.concat(ilogits),
            "correct": torch.concat(correct, 1),
        },
        f"/ssd1/tta/{prefix}_resnet50_bn_INC0-5.pth",
    )
