import torch
from torch import nn
import torch.nn.functional as F
import os

__all__ = ['HRNet', 'hrnetv2_48', 'hrnetv2_32']

# Checkpoint path of pre-trained backbone (edit to your path). Download backbone pretrained model hrnetv2-32 @
# https://drive.google.com/file/d/1NxCK7Zgn5PmeS7W1jYLt5J9E0RRZ2oyF/view?usp=sharing .Personally, I added the backbone
# weights to the folder /checkpoints

model_urls = {
    'hrnetv2_32': './checkpoints/model_best_epoch96_edit.pth',
    'hrnetv2_48': None
}


def check_pth(arch):
    CKPT_PATH = model_urls[arch]
    if os.path.exists(CKPT_PATH):
        print(f"Backbone HRNet Pretrained weights at: {CKPT_PATH}, only usable for HRNetv2-32")
    else:
        print("No backbone checkpoint found for HRNetv2, please set pretrained=False when calling model")
    return CKPT_PATH
    # HRNetv2-48 not available yet, but you can train the whole model from scratch.


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
