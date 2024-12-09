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

        # FNPLayer.register_to(self.model.layer1, dim=256, sample_n=2)
        # FNPPlusLayer.register_to(self.model.layer2, dim=512, sample_n=1)
        FNPPlusLayer.register_to(self.model.blocks[-2], dim=768, sample_n=1, plus=True)
        # FNPLayer.register_to(self.model.layer3, dim=1024, sample_n=0.5)

        params, param_names = collect_params(model)
        self.optimizer = torch.optim.SGD(params, lr=0.00025, momentum=0.9)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)


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
    
    # u = augment(x)
    # output_aug = model(u)
    # _, _, z2, p2 = model.layer1._outputs
    # loss_noise = (-F.cosine_similarity(p1, z2.detach()).mean(0) \
    #             + -F.cosine_similarity(p2, z1.detach()).mean(0)) * 0.5
    
    # logit = outputs.softmax(1)
    # logit_aug = output_aug.softmax(1)
    # cls1 = logit.argmax(dim=1)

    # plpd = torch.gather(logit, dim=1, index=cls1.reshape(-1,1)) - torch.gather(logit_aug, dim=1, index=cls1.reshape(-1,1))
    # plpd = plpd.reshape(-1)
    # loss_plpd = -plpd.mean(0)
    # pred, pred_w = outputs.split(x.shape[0])
    B = x.shape[0]
    C = outputs.shape[1]

    

    pred = outputs[:B]
    pred_w = outputs[B:] #kB
    P = pred_w.shape[0]
    k = P//B

    ent = softmax_entropy(pred)
    idx = ent < np.log(C) * 0.4
    pred_w = pred_w.view(k, B, C)
    pred_wc = pred_w[:,idx].view(-1, C)

    aug_coeff = (3 / (ent[idx].clone().detach() - 0.4).exp())
    # aug_coeff = 1
    
    loss_ent = softmax_entropy(pred[idx]).mean(0)
    # loss_aug = F.cross_entropy(pred_w[idx], pred[idx].softmax(1).detach())
    loss_aug = F.cross_entropy(pred_wc, pred[idx].argmax(1).expand(k, -1).flatten().detach(), reduction='none')
    loss_aug = (loss_aug * aug_coeff).mean()
    # loss_aug = loss_aug[loss_aug>0].mean()
    # loss_aug = (1 - F.cosine_similarity(pred_w[idx], pred[idx], dim=1)).mean(0)
    # loss = 2*loss_noise + loss_plpd + loss_ent
    # loss = loss * 0.01

    global n_iter
    loss = loss_aug * 5 #* max(1, 20 - n_iter)
    n_iter += 1

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    wandb.log({
        'loss': loss,
        'loss/aug': loss_aug,
        # 'loss/orth': loss_orth,
        # 'loss/noise': loss_noise,
        'loss/ent': loss_ent,
        # 'loss/plpd': loss_plpd
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