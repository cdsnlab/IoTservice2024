# -*- coding: utf-8 -*-
import math
from collections import OrderedDict

import torch.nn as nn
from timm.models.helpers import checkpoint_seq
# from ttab.configs.datasets import dataset_defaults

dataset_defaults = {
    "cifar10": {
        "statistics": {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
            "n_classes": 10,
        },
        "version": "deterministic",
        "img_shape": (32, 32, 3),
    },
    "cifar100": {
        "statistics": {
            "mean": (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            "std": (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
            "n_classes": 100,
        },
        "version": "deterministic",
        "img_shape": (32, 32, 3),
    },
    "officehome": {
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 65,
        },
        "img_shape": (224, 224, 3),
    },
    "pacs": {
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 7,
        },
        "img_shape": (224, 224, 3),
    },
    "coloredmnist": {
        "statistics": {
            "mean": (0.1307, 0.1307, 0.0),
            "std": (0.3081, 0.3081, 0.3081),
            "n_classes": 2,
        },
        "img_shape": (28, 28, 3),
    },
    "waterbirds": {
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 2,
        },
        "group_counts": [3498, 184, 56, 1057],  # used to compute group ratio.
        "img_shape": (224, 224, 3),
    },
    "imagenet": {
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 1000,
        },
        "img_shape": (224, 224, 3),
    },
}

__all__ = ["resnet"]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def norm2d(group_norm_num_groups, planes):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        # group_norm_num_groups == planes -> InstanceNorm
        # group_norm_num_groups == 1 -> LayerNorm
        return nn.GroupNorm(group_norm_num_groups, planes)
    else:
        return nn.BatchNorm2d(planes)


class ViewFlatten(nn.Module):
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

        self.downsample = downsample
        self.stride = stride

        # some stats
        self.nn_mass = in_planes + out_planes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out.expand_as(residual) + residual
        out = self.relu(out)

        return out

