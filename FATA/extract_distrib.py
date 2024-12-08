import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import models.Res as Resnet
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


def layer_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
    module._style_mean = output.mean(
        (-1, -2)
    ).detach()  # style: [B C] <- output: [B C H W]


def extract(net, loader):
    l4 = []
    e1k = []
    pred = []

    net.eval()
    for i, (x, _) in enumerate(tqdm(loader)):
        x = x.cuda()

        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)

        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.layer4(x)
        l4.append(x.detach().cpu())

        x = net.avgpool(x)
        x = x.reshape(x.size(0), -1)
        e1k.append(x.detach().cpu())

        x = net.fc(x)
        pred.append(x.detach().cpu())

    l4 = torch.concat(l4)
    e1k = torch.concat(e1k)
    pred = torch.concat(pred)

    print(f"{l4.shape=}, {e1k.shape=}, {pred.shape=}")

    torch.save(l4, "/ssd1/tta/imagenet_val_resnet50_l4.pth")
    torch.save(e1k, "/ssd1/tta/imagenet_val_resnet50_e1k.pth")
    torch.save(pred, "/ssd1/tta/imagenet_val_resnet50_pred.pth")
