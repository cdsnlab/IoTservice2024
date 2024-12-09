# https://github.com/DequanWang/tent/blob/master/tent.py

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.utils.data

import wandb
import numpy as np
from models.Res import ResNet
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import math

import torchvision
import torchvision.transforms as T
from einops import rearrange
from .aug import *

# Tent++
class TentX(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model: ResNet, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        if hasattr(self.model, 'layer3'):
            # FNPPlusLayer.register_to(self.model.maxpool, dim=64, sample_n=1, plus=True, sigma=1.)
            # FNPPlusLayer.register_to(self.model.layer1, dim=256, sample_n=1, plus=True, sigma=1.)
            # FNPPlusLayer.register_to(self.model.layer2, dim=512, sample_n=1, plus=True, sigma=1.)
            FNPPlusLayer.register_to(self.model.layer3, dim=1024, sample_n=1, plus=True, sigma=1.)
        elif hasattr(self.model, 'blocks'):
            FNPPlusLayer.register_to(self.model.blocks[-2], dim=768, sample_n=1, plus=True, sigma=1.)
        else:
            raise NotImplementedError("FNP requires a model with layer3 or blocks[-2]")

        self.model.shead = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.Tanh()
        )
        self.model.shead[0].weight.data.add_(1)
        
        params, param_names = collect_params(model)
        params += self.model.shead.parameters()
        self.optimizer = torch.optim.SGD(params, lr=0.00025, momentum=0.9)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

        # self.augment = nn.Linear(1024, 1024, bias=True)

    def forward(self, x: torch.Tensor):
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs


    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

n_iter = 0

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x: torch.Tensor, model: ResNet, optimizer):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    outputs = model(x)#, return_feature=True)
    B = x.shape[0]
    C = outputs.shape[1]

    pred = outputs[:B]
    pred_w = outputs[B:] #kB
    P = pred_w.shape[0]
    k = P//B

    entropys = softmax_entropy(pred)
    loss_ent = entropys.mean()

    _ent = entropys.clone().detach()
    idx = torch.where(_ent < 0.5*math.log(1000))

    
    pred_w = pred_w[idx]
    # pred_w = model.shead(pred_w)

    if len(pred[idx]) > 0:
        ccls = pred[idx].exp().detach()
        loss_aug = F.cross_entropy(pred_w, ccls, reduction='none')
        # loss_aug = (F.cross_entropy(pred_w, ccls, reduction='none')
        #             + F.cross_entropy(ccls, pred_w, reduction='none')
        #             ) / 2  

        ent_marg = 0.4 * math.log(1000)
        aug_coeff = (1 / (_ent[idx] - ent_marg).exp())
        # aug_coeff = 1.
        loss_aug = loss_aug.mul(aug_coeff).mean()
    else:
        loss_aug = 0

    # if loss_aug != 0:
    loss = loss_aug + loss_ent
    
    if loss != 0:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    wandb.log({
        'loss': loss,
        'loss/aug': loss_aug,
        'loss/ent': loss_ent
    }, commit=False)
    return pred



def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():

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

        if isinstance(m, FAugLayer):
            for np, p in m.named_parameters():
                params.append(p)
                names.append(f"{nm}.{np}")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"