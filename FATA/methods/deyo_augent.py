"""
Copyright to DeYO Authors, ICLR 2024 Spotlight (top-5% of the submissions)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torchvision
import torchvision.transforms as T
import math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import wandb
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from .aug import *

class DeYOAug(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model: nn.Module, args, optimizer, steps=1, episodic=False, deyo_margin=0.5*math.log(1000), margin_e0=0.4*math.log(1000)):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.args = args
        if args.wandb_log:
            import wandb
        self.steps = steps
        self.episodic = episodic
        args.counts = [1e-6,1e-6,1e-6,1e-6]
        args.correct_counts = [0,0,0,0]

        if hasattr(self.model, 'layer3'):
            FNPPlusLayer.register_to(self.model.layer3, dim=1024, sample_n=1, plus=True, sigma=1.)
            # FNPPlusLayer.register_to(self.model.layer4, dim=2048, sample_n=1, plus=True, sigma=1.)
            # FNPPlusLayer.register_to(self.model.layer2, dim=512, sample_n=1, plus=True, sigma=1.)
        elif hasattr(self.model, 'blocks'):
            FNPPlusLayer.register_to(self.model.blocks[-2], dim=768, sample_n=1, plus=True, sigma=1.)
            pass
        else:
            raise NotImplementedError("Model not supported for DeYOAug")

        self.deyo_margin = deyo_margin
        self.margin_e0 = margin_e0
    

    def forward(self, x, iter_, targets=None, flag=True, group=None):
        if self.episodic:
            self.reset()
        
        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward = forward_and_adapt_sar(x, iter_, self.model, self.args,
                                                                              self.optimizer, self.deyo_margin,
                                                                              self.margin_e0, targets, flag, group)
                else:
                    outputs = forward_and_adapt_sar(x, iter_, self.model, self.args,
                                                    self.optimizer, self.deyo_margin,
                                                    self.margin_e0, targets, flag, group)
        else:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = forward_and_adapt_sar(x, iter_, self.model, 
                                                                                                    self.args, 
                                                                                                    self.optimizer, 
                                                                                                    self.deyo_margin,
                                                                                                    self.margin_e0,
                                                                                                    targets, flag, group)
                else:
                    outputs = forward_and_adapt_sar(x, iter_, self.model, 
                                                    self.args, self.optimizer, 
                                                    self.deyo_margin,
                                                    self.margin_e0,
                                                    targets, flag, group)
        if targets is None:
            if flag:
                return outputs, backward, final_backward
            else:
                return outputs
        else:
            if flag:
                return outputs, backward, final_backward, corr_pl_1, corr_pl_2
            else:
                return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

# @torch.enable_grad()
# def aug_loss(x, outputs):
#     B = x.shape[0]
#     C = outputs.shape[1]
#     pred = outputs[:B]
#     pred_w = outputs[B:] #kB
#     P = pred_w.shape[0]
#     k = P//B

#     pred_w = pred_w.view(k, B, C)
#     pred_wc = pred_w[:,idx].view(-1, C)
#     loss_aug = F.cross_entropy(pred_wc, pred[idx].argmax(1).expand(k, -1).flatten())
#     return loss_aug

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


n_iter = 0
@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_sar(x, iter_, model, args, optimizer, deyo_margin, margin, targets=None, flag=True, group=None):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    aug_layer = model.blocks[-2] if hasattr(model, 'blocks') else model.layer3
    faug_layer = aug_layer._aug_layer

    B = x.size(0)
    faug_layer.enable()
    outputs = model(x)
    faug_layer.disable()


    pred_w = outputs[B:]
    outputs = outputs[:B]
    
    
    # print(f'{outputs.shape=}, {pred_w.shape=}')
    # pred_o = outputs[(augment_n+1)*B:]
    # is_augmented = ((pred_w[:B].view/(B, -1) == outputs.view(B, -1)).type(torch.float).mean(1) < 1)
    if not flag:
        return outputs
    


    entropys = softmax_entropy(outputs)
    wandb.log({'loss/ent': entropys.mean(0)}, commit=False)
    if args.filter_ent:
        filter_ids_1 = torch.where((entropys < deyo_margin))
    else:    
        filter_ids_1 = torch.where((entropys <= math.log(1000)))
    entropys = entropys[filter_ids_1]
    wandb.log({'loss/ent_filtd': entropys.mean(0)}, commit=False)
    
    backward = len(entropys)
    if backward==0:
        if targets is not None:
            return outputs, 0, 0, 0, 0
        return outputs, 0, 0

    x_prime = x[filter_ids_1]
    x_prime = x_prime.detach()
    if args.aug_type=='occ':
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(-1, -1, args.occlusion_size, args.occlusion_size)
        x_prime[:, :, args.row_start:args.row_start+args.occlusion_size,args.column_start:args.column_start+args.occlusion_size] = occlusion_window
    elif args.aug_type=='patch':
        resize_t = T.Resize(((x.shape[-1]//args.patch_len)*args.patch_len,(x.shape[-1]//args.patch_len)*args.patch_len))
        resize_o = T.Resize((x.shape[-1],x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.patch_len, ps2=args.patch_len)
        # perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
        perm_idx = torch.argsort(
            torch.Tensor(np.random.rand(x_prime.shape[0],x_prime.shape[1])), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
        x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.patch_len, ps2=args.patch_len)
        x_prime = resize_o(x_prime)
    elif args.aug_type=='pixel':
        x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
        x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])

    with torch.no_grad():
        faug_layer.disable()
        outputs_prime = model(x_prime)
        faug_layer.enable()
    outputs_prime = outputs_prime[:x_prime.size(0)]
    
    # if hasattr(model, 'layer3'):
    #     for lyr in [model.layer1, model.layer2, model.layer3]:
    #         if hasattr(lyr, '_aug_layer'):
    #             lyr._aug_layer.enable()
    
    prob_outputs = outputs[filter_ids_1].softmax(1)
    prob_outputs_prime = outputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)
    wandb.log({'loss/plpd': plpd.mean(0)}, commit=False)
    
    if args.filter_plpd:
        filter_ids_2 = torch.where(plpd > args.plpd_threshold)
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)
    entropys = entropys[filter_ids_2]
    final_backward = len(entropys)
    plpd = plpd[filter_ids_2]
    wandb.log({'loss/plpd_filtd': plpd.mean(0)}, commit=False)
    
    if targets is not None:
        corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()
        corr_pl_2 = (targets[filter_ids_1][filter_ids_2] == prob_outputs[filter_ids_2].argmax(dim=1)).sum().item()

    if args.reweight_ent or args.reweight_plpd:
        coeff = (args.reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                 args.reweight_plpd * (1 / (torch.exp(-1. * plpd.clone().detach())))
                )            
        entropys = entropys.mul(coeff)
        
    loss_ent = entropys.mean(0)

    # print(f'{filter_ids_1=}\n{filter_ids_1[0].shape=}\n{is_augmented.shape=} ')
    # filt_ent = torch.zeros_like(is_augmented).type(torch.bool)
    # filt_ent[filter_ids_1] = True

    
    _ent = softmax_entropy(outputs).clone().detach()

    idx = torch.where(_ent < 0.5*math.log(1000))

    # filt_ent = torch.zeros_like(_ent).type(torch.bool)
    # filt_ent[idx] = True

    # filt_plpd = torch.zeros_like(_ent).type(torch.bool)
    # filt_plpd[filter_ids_2] = True

    # idx = filt_ent & filt_plpd
    pred_w = pred_w[idx]
    # pred_w = pred_w[filter_ids_2]
    
    P = pred_w.shape[0]
    k = P//B
    loss_aug = 0
    if P > 0:
        # ccls = outputs[idx].argmax(dim=1).flatten().detach()
        loss_aug = F.cross_entropy(pred_w, outputs[idx], reduction='none')

        ent_marg = 0.4 * math.log(1000)
        aug_coeff = (1 / (_ent[idx] - ent_marg).exp())
        # print(aug_coeff.max())
        # aug_coeff = 1.
        loss_aug = loss_aug.mul(aug_coeff).mean()


    loss = loss_ent + loss_aug # + loss_old #ResNet50/
    wandb.log({'loss/ent_final': loss_ent,
               'loss/aug': loss_aug,
            #    'loss/old': loss_old,
               'loss': loss}, commit=False)

    if final_backward != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()

    del x_prime
    del plpd
    
    if targets is not None:
        return outputs, backward, final_backward, corr_pl_1, corr_pl_2
    return outputs, backward, final_backward

def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with DeYO."""
    # train mode, because SAR optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what SAR updates
    model.requires_grad_(False)
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model

