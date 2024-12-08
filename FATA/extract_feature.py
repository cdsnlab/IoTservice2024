import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import models.Res as Resnet
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


def extract(net, loader):
    l3 = []

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
        l3.append(x.detach().cpu())

        if i >= 1000:
            break

    l3 = torch.concat(l3)

    torch.save(l3, "/ssd1/tta/inc_val_resnet50_l3.pth")


def main():
    net = Resnet.__dict__["resnet50"](pretrained=True)
    net = net.cuda()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ]
    )

    dataset = ImageFolder("/ssd1/datasets/ImageNet/val", transform=transform)
    loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=16)

    extract(net, loader)


if __name__ == "__main__":
    with torch.no_grad():
        main()
