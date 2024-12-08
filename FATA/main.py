"""
Copyright to DeYO Authors
built upon on Tent, EATA, and SAR code.
"""

import os
import time
import math
from config import get_opts

import random
import wandb
from datetime import datetime

import numpy as np
from utils.utils import get_logger
from dataset.selectedRotateImageFolder import prepare_test_data
from utils.cli_utils import *

from methods.aug import FNPPlusLayer

import torch

from methods import *
import timm

import models.Res as Resnet

from tqdm import tqdm


def validate(val_loader, model, args):
    logger = args.logger

    acc1s, acc5s = [], []

    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    val_loader = tqdm(val_loader)
    val_loader.set_description(args.corruption)

    if args.method.startswith("deyo"):
        count_backward = 1e-6
        final_count_backward = 1e-6
        count_corr_pl_1 = 0
        count_corr_pl_2 = 0
        total_count_backward = 1e-6
        total_final_count_backward = 1e-6
        total_count_corr_pl_1 = 0
        total_count_corr_pl_2 = 0
    model.eval()

    global_step = 0
    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            group = None

            if args.method.startswith("deyo"):
                output, backward, final_backward, corr_pl_1, corr_pl_2 = model(
                    images, i, target, group=group
                )
                count_backward += backward
                final_count_backward += final_backward
                total_count_backward += backward
                total_final_count_backward += final_backward

                count_corr_pl_1 += corr_pl_1
                count_corr_pl_2 += corr_pl_2
                total_count_corr_pl_1 += corr_pl_1
                total_count_corr_pl_2 += corr_pl_2
            else:
                output = model(images)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if True:
                if args.wandb_log:
                    wandb.log(
                        {
                            "iter": i,
                            "gstep": global_step,
                            f"top1/{args.corruption}": top1.avg,
                            f"batch/{args.corruption}": acc1,
                            "accuracy": top1.avg,
                        },
                        commit=True,
                    )

            val_loader.set_postfix(dict(top1=top1, top5=top5))

            batch_time.update(time.time() - end)
            end = time.time()

            global_step += 1

    logger.info(
        f"Result under {args.corruption}. The adaptation accuracy of {args.method} is top1: {top1.avg:.5f} and top5: {top5.avg:.5f}"
    )

    acc1s.append(top1.avg.item())
    acc5s.append(top5.avg.item())

    logger.info(f"acc1s are {acc1s}")
    logger.info(f"acc5s are {acc5s}")
    return top1.avg, top5.avg


def run_tent(args, net, logger):
    net = tent.configure_model(net)
    params, param_names = tent.collect_params(net)
    optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
    tented_model = tent.Tent(net, optimizer)
    return tented_model


def def_run_tentx(module_name):
    if module_name not in globals():
        raise NotImplementedError(f"module {module_name} is not implemented.")

    module = globals()[module_name]

    def _run_tentx(args, net, logger):
        net = module.configure_model(net)
        params, param_names = module.collect_params(net)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        tented_model = module.TentX(net)
        return tented_model

    return _run_tentx


def def_run_deyox(module_name):
    if module_name not in globals():
        raise NotImplementedError(f"module {module_name} is not implemented.")

    module = globals()[module_name]

    def _run_deyo_aug(args, net, logger):
        net = module.configure_model(net)
        params, param_names = module.collect_params(net)

        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        adapt_model = module.DeYOAug(
            net,
            args,
            optimizer,
            deyo_margin=args.deyo_margin,
            margin_e0=args.deyo_margin_e0,
        )

        return adapt_model

    return _run_deyo_aug


def run_tentb(args, net, logger):
    net = tentb.configure_model(net)
    params, param_names = tentb.collect_params(net)
    optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
    tented_model = tentb.TentB(net, optimizer)
    return tented_model


def run_stent(args, net, logger):
    net = stent.configure_model(net)
    params, param_names = stent.collect_params(net)
    optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
    tented_model = stent.STent(net)
    return tented_model


def run_otent(args, net, logger):
    src_l1ft = torch.load("/ssd1/tta/imagenet_val_resnet50_lyrfts.pth")

    net = ot.configure_model(net)
    params, param_names = ot.collect_params(net)
    optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
    tented_model = ot.OTent(net, optimizer, src_l1ft)
    return tented_model


def run_eata(args, net, logger, aug=False):
    net_ewc = Resnet.__dict__["resnet50"](pretrained=True)
    net_ewc = net_ewc.cuda()

    net_ewc = eata.configure_model(net_ewc)
    params, param_names = eata.collect_params(net_ewc)
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}

    corruption = args.corruption
    if args.eata_fishers:
        print("EATA!")
        args.corruption = "original"

        fisher_dataset, fisher_loader = prepare_test_data(args)
        fisher_dataset.set_dataset_size(args.fisher_size)
        fisher_dataset.switch_mode(True, False)

        net = eata.configure_model(net)
        params, param_names = eata.collect_params(net)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, data in enumerate(fisher_loader, start=1):
            images, targets = data[0], data[1]
            if args.gpu is not None:
                images = images.cuda(non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(non_blocking=True)
            outputs = net(images)
            if aug:
                outputs = outputs
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in net.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = (
                            param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        )
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        logger.info("compute fisher matrices finished")
        del ewc_optimizer
    else:
        net = eata.configure_model(net)
        params, param_names = eata.collect_params(net)
        print("ETA!")
        fishers = None

    args.corruption = corruption
    optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)

    if not aug:
        adapt_model = eata.EATA(
            args,
            net,
            optimizer,
            fishers,
            args.fisher_alpha,
            e_margin=args.e_margin,
            d_margin=args.d_margin,
        )
    else:
        adapt_model = eata_aug.EATAAug(
            args,
            net,
            optimizer,
            fishers,
            args.fisher_alpha,
            e_margin=args.e_margin,
            d_margin=args.d_margin,
        )

    return adapt_model


def run_sar(args, net, logger, aug=False):
    net = sar.configure_model(net)
    params, param_names = sar.collect_params(net)

    base_optimizer = torch.optim.SGD
    optimizer = sam.SAM(params, base_optimizer, lr=args.lr, momentum=0.9)

    if not aug:
        adapt_model = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)
    else:
        adapt_model = sar_aug.SARAug(net, optimizer, margin_e0=args.sar_margin_e0)

    return adapt_model


def run_deyo(args, net, logger):
    net = deyo.configure_model(net)
    params, param_names = deyo.collect_params(net)

    optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
    adapt_model = deyo.DeYO(
        net,
        args,
        optimizer,
        deyo_margin=args.deyo_margin,
        margin_e0=args.deyo_margin_e0,
    )

    return adapt_model


def run_deyo_aug(args, net, logger):
    net = deyo_aug.configure_model(net)
    params, param_names = deyo_aug.collect_params(net)

    optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
    adapt_model = deyo_aug.DeYOAug(
        net,
        args,
        optimizer,
        deyo_margin=args.deyo_margin,
        margin_e0=args.deyo_margin_e0,
    )

    return adapt_model
