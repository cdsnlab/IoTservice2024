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

    # print(f"{l4.shape=}, {e1k.shape=}, {pred.shape=}")

    torch.save(l4, "/ssd1/tta/imagenet_val_resnet50_l4.pth")
    torch.save(e1k, "/ssd1/tta/imagenet_val_resnet50_e1k.pth")
    torch.save(pred, "/ssd1/tta/imagenet_val_resnet50_pred.pth")


def extract_clsft(net, loader):
    net.eval()

    fts = {i: [] for i in range(1000)}
    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.cuda()

        ft = net(x, return_feature=True, return_feature_only=True)

        for f, t in zip(ft, y):
            fts[t.item()].append(f.detach().cpu())

    results = {k: torch.concat(v) for k, v in fts.items()}

    torch.save(results, "/ssd1/tta/imagenet_val_resnet50_clsfts.pth")
    print(f"{fts[0].shape=}")


def extract_lyrft(net, loader):
    net.eval()

    target_layers = [net.layer1, net.layer2, net.layer3, net.layer4]
    fts = [[] for _ in target_layers]
    for i, (x, _) in enumerate(tqdm(loader)):
        x = x.cuda()
        x = net(x)

        for j, layer in enumerate(target_layers):
            fts[j].append(layer._style_mean.cpu())

    results = [torch.concat(f) for f in fts]
    print([f.shape for f in results])

    torch.save(results, "/ssd1/tta/imagenet_val_resnet50_lyrfts.pth")


def main():
    net = Resnet.__dict__["resnet50"](pretrained=True)
    net = net.cuda()
    net.layer1.register_forward_hook(layer_hook)
    net.layer2.register_forward_hook(layer_hook)
    net.layer3.register_forward_hook(layer_hook)
    net.layer4.register_forward_hook(layer_hook)

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

    # extract_clsft(net, loader)
    extract_lyrft(net, loader)


if __name__ == "__main__":
    with torch.no_grad():
        main()
