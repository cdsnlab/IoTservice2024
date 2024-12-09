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

import torchvision
import torchvision.transforms as T
from einops import rearrange

def mahalanobis(u, distrib, reduction='mean'):
    v, cov_inv = distrib #C, C
    delta = (u - v) #B, C
    t = delta @ cov_inv #B, C
    m = t.unsqueeze(1) @ delta.unsqueeze(-1) #B
    # m = torch.dot(delta, torch.matmul(cov_inv.double(), delta.T))
    return torch.sqrt(m).mean().float()

def layer_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
    module._style_mean = output.mean((-1, -2)) #style: [B C] <- output: [B C H W]

def rcrop(x: torch.Tensor, low: float, high: float=1.) :
    rx = torch.randint(int(x.shape[-2]*low), int(x.shape[-2]*high), (1,))
    ry = torch.randint(int(x.shape[-1]*low), int(x.shape[-1]*high), (1,))
    crop = T.RandomCrop((int(rx), int(ry)))
    resize = T.Resize((x.shape[-2], x.shape[-1]))
    x_crop = resize(crop(x))
    return x_crop

# structure-destructing augmentation
def augment(x: torch.Tensor, patch_len=4):
    resize_t = T.Resize(((x.shape[-1]//patch_len)*patch_len,(x.shape[-1]//patch_len)*patch_len))
    resize_o = T.Resize((x.shape[-1],x.shape[-1]))

    x_prime = resize_t(x)
    x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=patch_len, ps2=patch_len)
    perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
    x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
    x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=patch_len, ps2=patch_len)
    x_prime = resize_o(x_prime)
    return x_prime


class Refiner(nn.Module):
    def __init__(self, dim) -> None:
        super(Refiner, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, affine=False)
        )
        # self.project = nn.Sequential(
        #     nn.Linear(dim, dim, bias=False),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dim, dim),
        #     nn.BatchNorm1d(dim, affine=False)
        # )
        self.predict = nn.Sequential(
            nn.Linear(dim, dim//2, bias=False),
            nn.BatchNorm1d(dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(dim//2, dim)
        )

        # self.project = nn.Sequential(
        #     nn.Linear(dim, dim//4, bias=False),
        #     nn.BatchNorm1d(dim//4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dim//4, dim)
        # )
        
    def forward(self, input: torch.Tensor):
        B, C, H, W = input.size()
        x = rearrange(input, 'b c h w -> (b h w) c')
        z = self.project(x)
        p = self.predict(z)

        r = x - z
        dot = torch.sum(r * z, dim=1).pow(2).mean(0).sqrt() #torch.dot(r, n)
        r = rearrange(r, '(b h w) c -> b c h w', b=B, h=H, w=W)
        return r, dot, z, p
    
        # n = self.project(x)
        # r = x - n
        # dot = torch.sum(r * n, dim=1).pow(2).mean(0).sqrt() #torch.dot(r, n)
        # r = rearrange(r, '(b h w) c -> b c h w', b=B, h=H, w=W)
        # return r, dot
        


def layer_drop_hook(refine):
    def _layer_drop_hook(module: nn.Module, args, output: torch.Tensor):
        r, dot, z, p = refine(output) #b c h w
        module._outputs = (output, r, z, p)
        module._noise = output - r
        module._loss_orth = dot
        return r
    return _layer_drop_hook
        
# Tent++
class STent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model: ResNet, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        
        refine_layers = 1
        self.refine = nn.ModuleList([
            Refiner(d) for i, d in enumerate([256, 512, 1024, 2048][:refine_layers])
        ])
        for i, layer in enumerate([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4][:refine_layers]):
            layer.register_forward_hook(layer_drop_hook(self.refine[i]))
        
        # self.model.layer1.register_forward_hook(layer_hook)
        # self.model.layer2.register_forward_hook(layer_hook)
        # self.model.layer3.register_forward_hook(layer_hook)
        # self.model.layer4.register_forward_hook(layer_hook)

        params, param_names = collect_params(model)
        params += self.refine.parameters()
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


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x: torch.Tensor, model: ResNet, optimizer):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    outputs, x_emb = model(x, return_feature=True)
    _, _, z1, p1 = model.layer1._outputs
    loss_orth = model.layer1._loss_orth
    # in_n = rearrange(model.layer1._noise, 'b c h w -> (b h w) c')
    # in_ft = rearrange(model.layer1._output, 'b c h w -> (b h w) c').detach()
    
    u = augment(x)
    output_aug = model(u)
    _, _, z2, p2 = model.layer1._outputs
    loss_noise = (-F.cosine_similarity(p1, z2.detach()).mean(0) \
                + -F.cosine_similarity(p2, z1.detach()).mean(0)) * 0.5

    # noise_n = rearrange(model.layer1._noise, 'b c h w -> (b h w) c')
    # noise_ft = rearrange(model.layer1._output, 'b c h w -> (b h w) c')#.detach()

    # loss_noise = (1 - F.cosine_similarity(in_n, noise_n)).mean(0) \
    #             + F.mse_loss(noise_n, noise_ft).mean(0)
    
    logit = outputs.softmax(1)
    logit_aug = output_aug.softmax(1)
    cls1 = logit.argmax(dim=1)

    plpd = torch.gather(logit, dim=1, index=cls1.reshape(-1,1)) - torch.gather(logit_aug, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)
    loss_plpd = -plpd.mean(0)

    loss_ent = softmax_entropy(outputs).mean(0)
    loss = loss_orth + 2*loss_noise + loss_plpd + loss_ent
    # loss = loss * 0.01

    # loss = loss_mh + loss_ent
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    wandb.log({
        'loss': loss,
        'loss/orth': loss_orth,
        'loss/noise': loss_noise,
        'loss/ent': loss_ent,
        'loss/plpd': loss_plpd
    }, commit=False)
    return outputs



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
            
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
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