import torch
import torch.nn as nn
import torch.random
from einops import rearrange
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Uniform

import wandb

def rearranger(pattern):
    def _fn(x, **kwargs):
        return rearrange(x, pattern, **kwargs)
    return _fn

expand_4d = rearranger('b c -> b c 1 1')

class EMA:
    def __init__(self, p=0.9):
        self.value = None
        self.p = p
    
    def update(self, value):
        self.value = value \
            if self.value is None \
            else self.p * self.value.detach() + (1 - self.p) * value 
        return self.value
    
    def get(self):
        return self.value
    

class FAugLayer():
    def __init__(self, dim: int, half=False) -> None:
        super().__init__()
        self.dim = dim
        self.is_enabled = True
        self.plabel = None
        self.half = half

    def disable(self):
        self.is_enabled = False
        self.plabel = None
    
    def enable(self, plabel=None):     
        self.is_enabled = True
        self.plabel = plabel
    
    def _augment(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        if not self.is_enabled: return x
        t = x
        if self.half:
            t = x.split(x.size(0) // 2)[1]
        y = self._augment(t, self.plabel)
        # self._input = x.clone().detach()
        # self._output = y.clone().detach()
        return torch.cat((x, y), dim=0)
    
    def hook(self, module: nn.Module, args, output: torch.Tensor):
        if not self.is_enabled:
            return output
        module._output = output
        # module._aug = self.forward(output)
        return self.forward(output)
    
    @classmethod
    def register_to(cls, layer: nn.Module|str, **kwargs):
        aug_layer = cls(**kwargs)        
        # aug_layer._target_layer = layer
        layer.register_forward_hook(aug_layer.hook)
        layer._aug_layer = aug_layer
        return layer   


class ProjectorLayer(nn.Module):
    def __init__(self, dim, bias=True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(dim, requires_grad=True)) if bias else None
        self._init_weights()
    
    def _init_weights(self):
        self.weight.data.fill_(1)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x):
        x = self.weight * x
        if self.bias is not None:
            x = x + self.bias
        return x

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) >= 2)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt()
    feat_mean = feat.view(N, C, -1).mean(dim=2)
    return feat_mean, feat_std

class FNPPlusLayer(FAugLayer):
    def __init__(self, dim: int, sigma: float=1., sample_n: int|float=1, batch_size: int=64, plus=True, half=False) -> None:
        super().__init__(dim, half)
        self.sigma = sigma
        self.batch_size = batch_size
        self.sample_n = sample_n
        self.plus = plus
        self.dist_a = MultivariateNormal(torch.ones(dim, requires_grad=False), self.sigma * torch.eye(dim, requires_grad=False))
        self.var_a = EMA(p=0.95)
        self.dist_s = MultivariateNormal(torch.zeros(dim, requires_grad=False), 0.1 * torch.eye(dim, requires_grad=False))
        # self.project_a = ProjectorLayer(dim, bias=False).cuda()
        # self.project_b = ProjectorLayer(dim, bias=False).cuda()
        # self.project = nn.ModuleList([self.project_a, self.project_b])
        self.src_stat = torch.load('/ssd1/tta/imagenet_val_resnet50_lyr3_stat.pth').cuda()


    def _augment(self, x: torch.Tensor, plabel: torch.Tensor = None):
        if not self.is_enabled: return x
    
        D = len(x.shape)
        assert D == 3 or D == 4 
        N = x.size(0)
        k = max(self.sample_n, 1)

        if D == 3:
            x = x.permute(0, 2, 1) # B, C, L
        
        if D == 4:
            mu_c = x.mean((-1, -2)) # B, C
        elif D == 3:
            mu_c = x.mean(-1) # B, C, L -> B, C
        

        mu_c = mu_c.repeat(k, 1) # kB, C
    
        alpha = self.dist_a.sample((k*N,)).to(x.device).detach()  # kB, C
        beta = self.dist_a.sample((k*N,)).to(x.device).detach()  # kB, C

        if D == 4:
            x = x.repeat(k, 1, 1, 1)
            y = expand_4d(alpha) * x.repeat(k, 1, 1, 1) + expand_4d((beta - alpha) * mu_c) #kB, C, H, W
        elif D == 3:
            y = alpha.unsqueeze(-1) * x.repeat(k, 1, 1) + ((beta - alpha) * mu_c).unsqueeze(-1) #kB, C, L
            y = y.permute(0, 2, 1)
        
        if self.sample_n < 1:
            n = int(N * self.sample_n)
            i = torch.randperm(N)[:n]
            y[i] = x[i]

        return y


class FNPLayer(FNPPlusLayer):
    def __init__(self, dim: int, sigma: float=1., sample_n: int|float=1, batch_size: int=64) -> None:
        super().__init__(dim, sigma, sample_n, batch_size, plus=False)