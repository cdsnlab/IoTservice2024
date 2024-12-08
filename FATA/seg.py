mport os
import random
import json
import pickle
import yaml
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import network
from datasets.utils import get_acdc_continual_dataloader, get_cityscapes_c_continual_dataloader
from tqdm import tqdm
from config import get_opts, opts_to_dict, dump_opts
from metrics import StreamSegMetrics
from network import get_segmentation_model
from network.adaptbn import apply_adaptBN
from network.utils import deeplabv3plus_resnet50
from tools import softmax_entropy
from tools.setup import setup_segmentation
from typing import Dict
from datasets.evaluator import CityscapeEvaluator
import methods.tent

from easydict import EasyDict

import wandb

from methods.aug import *

from transformers import SegformerForSemanticSegmentation

@torch.jit.script
def flat2d(x: torch.Tensor):
    return x.view(x.shape[0], -1)

def main(opts: Dict[str, any]):
    assert opts.exp_name is not None

    flag_transfer = True
    flag_backward =  True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    input_size = None# (540, 960)
    
    # if opts.data == 'cityscapes-c':
    #     train_loader = get_cityscapes_c_continual_dataloader(opts, 'val', 'bfrs', resize=input_size)
    #     images_per_round = 500
    # else:
    #     train_loader = get_acdc_continual_dataloader(opts, split='train', order='fnrs', resize=input_size)
    #     images_per_round = 400
    
    # print(f"Loader: {len(train_loader)}, order: {opts.data_order}, rounds: {opts.rounds}")

    # tune_layers = opts.hparams.tune_layers


    cur_iter = 0
    log_dir = os.path.join('.', 'experiments', 'seg', opts.model, opts.exp_name)
    out_dir = os.path.join(log_dir, 'out')
    os.makedirs(out_dir, exist_ok=True)
    

    wandb.init(
        project=f"SS_{opts.model}",
        tags=['ideation']
    )
    wandb.run.name = f"{opts.exp_name}"
    wandb.define_metric('iter')
    wandb.define_metric('mIoU', step_metric="iter")


    print(f"Experiment '{opts.exp_name}' begins:")
    

    for case in 'fnrs':
        print('Case:', case)
        metrics = StreamSegMetrics(19)
        model = deeplabv3plus_resnet50(num_classes=19)
        model.load_state_dict(torch.load(opts.pretrained)['model_state'])
        model = model.cuda()
        model.eval()

        model.requires_grad_(False)
        methods.tent.configure_model(model.backbone)
        model.eval()

        FNPPlusLayer.register_to(model.backbone.layer3, dim=1024, sample_n=1, plus=True, sigma=1.)

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)

        metrics.reset()
        # model.eval()

        metrics_one = StreamSegMetrics(19)
        metrics_one.reset()

        train_loader = get_acdc_continual_dataloader(opts, split='train', order=case, resize=input_size)
        images_per_round = 400

        loader = tqdm(train_loader)
        for i,(img_id, tar_id, images, labels) in enumerate(loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            targets = labels.cpu().numpy()
            if i == 0:
                print(f"Input size = {images.shape}, Labels size = {labels.shape}")
            outputs, _ = model(images)

            B = images.shape[0]

            pred_w = outputs[B:] #kB
            outputs = outputs[:B]
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            P = pred_w.shape[0]

            # ft_w = ft[B:]

            outputs = F.avg_pool2d(outputs, kernel_size=(4,4))
            pred_w = F.avg_pool2d(pred_w, kernel_size=(4,4))

            outputs = outputs.permute(0, 2, 3, 1).reshape(-1, 19)
            pred_w = pred_w.permute(0, 2, 3, 1).reshape(-1, 19)

            entropys = softmax_entropy(outputs)
            loss_ent = entropys.mean()
            
            _ent = entropys.clone().detach()
            idx = torch.where(_ent < 0.5*math.log(1000))

            ccls = outputs[idx].argmax(dim=1).flatten().detach()
            loss_aug = F.cross_entropy(pred_w[idx], ccls, reduction='none')

            ent_marg = 0.4 * math.log(1000)
            aug_coeff = (1 / (_ent[idx] - ent_marg).exp())
            # aug_coeff = 1.
            loss_aug = loss_aug.mul(aug_coeff).mean()

            loss = loss_ent + loss_aug * 0.2

            if loss != 0 and opts.adapt:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            wandb.log({
                'loss': loss,
                'loss/aug': loss_aug,
                'loss/ent': loss_ent
            }, commit=False)

            metrics.update(targets, preds)

            metrics_one.reset()
            metrics_one.update(targets, preds)
            miou = metrics_one.get_results()["Mean IoU"]

            if cur_iter % 5 == 0:
                loader.set_postfix_str(f'{miou=:.4f} / {metrics.get_results()["Mean IoU"]:.4f}')

            wandb.log({
                f'mIoU': miou,
                'iter': cur_iter
            }, commit=True)
            cur_iter += 1 #opts.batch_size
        
        wandb.log({
            f'mIoU_{case}': metrics.get_results()["Mean IoU"],
            'iter': cur_iter
        }, commit=True)
            