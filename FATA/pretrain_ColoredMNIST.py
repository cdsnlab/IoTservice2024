import os
import argparse

parser = argparse.ArgumentParser(description="Waterbirds pretrain")
parser.add_argument("--root_dir", default=None, help="path to data")
parser.add_argument(
    "--dset_dir", default="ColoredMNIST", help="name of dataset directory"
)
parser.add_argument("--gpu", default="0", type=str, help="gpu index for training.")
parser.add_argument(
    "--seed", default=2024, type=int, help="seed for initializing training."
)
parser.add_argument(
    "--batch_size", default=64, type=int, help="batch_size for training."
)
parser.add_argument(
    "--test_batch_size", default=1000, type=int, help="batch_size for test."
)
parser.add_argument(
    "--workers", default=2, type=int, help="num_workers for train loader."
)
parser.add_argument("--if_shuffle", default=1, type=int, help="shuffle for training.")
parser.add_argument("--max_epochs", default=20, type=int, help="epochs for training.")
parser.add_argument("--interval", default=10, type=int, help="intervals for saving.")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import pickle

import matplotlib.pyplot as plt

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import torchvision
from torchvision import transforms
import torchvision.datasets.utils as dataset_utils

from dataset.ColoredMNIST_dataset import ColoredMNIST


def test_model(model, device, test_loader, set_name="test set"):
    model.eval()
    CELoss = torch.nn.CrossEntropyLoss()
    test_loss = 0
    correct_count = torch.tensor([0, 0, 0, 0])
    total_count = torch.tensor([0, 0, 0, 0])
    with torch.no_grad():
        for data, target, color in test_loader:
            data, target, color = data.to(device), target.to(device), color.to(device)
            group = 2 * target + color
            output = model(data)
            test_loss += CELoss(output, target).sum().item()  # sum up batch loss
            TFtensor = output.argmax(dim=1) == target
            for group_idx in range(4):
                correct_count[group_idx] += TFtensor[group_idx == group].sum().item()
                total_count[group_idx] += len(TFtensor[group_idx == group])

    test_loss /= len(test_loader.dataset)
    accs = correct_count / total_count * 100

    print(
        "\nPerformance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            set_name,
            test_loss,
            correct_count.sum().item(),
            total_count.sum().item(),
            correct_count.sum().item() / total_count.sum().item() * 100,
        )
    )
    print(
        "Group accuracy  => RSmall: {:.2f}, GSmall: {:.2f}, RBig: {:.2f}, GBig: {:.2f}".format(
            accs[0].item(), accs[1].item(), accs[2].item(), accs[3].item()
        )
    )
    print(
        "Detailed counts => RSmall: {}/{}, GSmall: {}/{}, RBig: {}/{}, GBig: {}/{}".format(
            correct_count[0],
            total_count[0],
            correct_count[1],
            total_count[1],
            correct_count[2],
            total_count[2],
            correct_count[3],
            total_count[3],
        )
    )

    return correct_count, total_count
