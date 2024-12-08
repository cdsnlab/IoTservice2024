import os
import argparse

parser = argparse.ArgumentParser(description="Waterbirds pretrain")
parser.add_argument("--root_dir", default=None, help="path to data")
parser.add_argument("--dset_dir", default=None, help="name of dataset directory")
parser.add_argument("--gpu", default="0", type=str, help="gpu index for training.")
parser.add_argument(
    "--seed", default=2024, type=int, help="seed for initializing training."
)
parser.add_argument(
    "--batch_size", default=64, type=int, help="batch_size for training."
)
parser.add_argument(
    "--test_batch_size", default=256, type=int, help="batch_size for test."
)
parser.add_argument(
    "--workers", default=2, type=int, help="num_workers for train loader."
)
parser.add_argument("--if_shuffle", default=1, type=int, help="shuffle for training.")
parser.add_argument("--max_epochs", default=200, type=int, help="epochs for training.")
parser.add_argument("--interval", default=10, type=int, help="intervals for saving.")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

SEED = args.seed
deterministic = True

import random
import torch
import numpy as np

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import wget
import tarfile
import h5py
import csv
from dataset.waterbirds_dataset import WaterbirdsDataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pickle
import models.Res as Resnet
from tqdm import tqdm


def download_dataset(root_dir):
    # url from Official github https://github.com/kohpangwei/group_DRO
    url = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
    filename = os.path.join(root_dir, "waterbird_complete95_forest2water2.tar.gz")
    if not os.path.isfile(filename):
        print("Downloading the tar file")
        wget.download(url, out=root_dir)
    else:
        print(filename, "already exists")
