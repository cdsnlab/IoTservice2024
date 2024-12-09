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

# def layer_drop_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
#     idx = torch.randint(256, (int(256 * 0.25),)).cuda()
#     output = output.index_fill(1, idx, 0)
#     return output

class Refiner(nn.Module):
    def __init__(self, dim, style) -> None:
        super(Refiner, self).__init__()
        # self.conv1 = nn.Conv2d(dim, dim*2, kernel_size=1)
        # self.dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2)
        # self.project = nn.Conv2d(dim, dim, kernel_size=1)
        # self.layer_norm = nn.LayerNorm(dim)
        self.style_mean = style['mean'].float().detach().cuda()
        self.style_std = style['cinv'].float().diag().detach().cuda()
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(dim, dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(dim, dim, bias=True),
            nn.Sigmoid()
            # nn.Tanh()
        )
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        # nn.init.eye_(self.classifier[0].weight)

        self.weight = nn.Sequential(
            nn.Linear(1, 1),
            nn.LeakyReLU(),
            nn.Linear(1, 1),
            nn.LeakyReLU(),
            nn.Linear(1, 1),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor):
        B, C, H, W = input.size()
        b = rearrange(input, 'b c h w -> c (b h w)')
        mean = b.mean(1).view(1, -1)
        var = b.var(1).view(1, -1)
        x = input.mean((-1, -2)) # B, C
        x = (x - mean) / (var + 1e-6).sqrt()
        x = self.classifier(x)
        x = self.conv(input * x.view(B, C, 1, 1))
        return x + input
        



        # # x = self.layer_norm(x)
        # # x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)\
        # dist = (x - self.style_mean) / self.style_std
        # # w = self.weight(dist.unsqueeze(-1)).view(B, C, 1, 1)
        # w = F.hardtanh(dist).view(B, C, 1,)
        # x = self.style_mean.view(1, -1, 1, 1) * w + (1-w) * input

        # # x = self.classifier(x) # B, C

        # # x = self.conv1(x)
        # # x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # # x = F.gelu(x1) * x2
        # # x = self.project(x)
        # # x = self.conv(input * x[:,:,None,None])
        # return x #+ input

def layer_drop_hook(refine):
    def _layer_drop_hook(module: nn.Module, args, output: torch.Tensor):
        r = refine(output)
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
        
        # self.src_style = torch.load('/ssd1/tta/imagenet_val_resnet50_shf_bn_full.pth')['ifeatures'][-1].mean(0).detach().cuda()
        # self.src_style = torch.load('/ssd1/tta/imagenet_val_resnet50_srcft.pth')[0].detach().cuda() # after block 1
        src_style = torch.load('/ssd1/tta/imagenet_val_resnet50_distributions.pth')
        # self.src_style = (src_style['mean'].detach().cuda(), src_style['cinv'].detach().cuda())

        refine_layers = 1
        self.refine = nn.ModuleList([
            Refiner(d, src_style[i]) for i, d in enumerate([256, 512, 1024, 2048][:refine_layers])
        ])
        # def _winit(m: nn.Conv2d):
        #     nn.init.zeros_(m.weight)
        #     nn.init.zeros_(m.bias)
        # self.restore.apply(_winit)

        # self.project[2].load_state_dict(deepcopy(self.model.fc.state_dict()))
        for i, layer in enumerate([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4][:refine_layers]):
            layer.register_forward_hook(layer_drop_hook(self.refine[i]))
        
        self.model.layer1.register_forward_hook(layer_hook)
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
    # loss_mh = mahalanobis(model.layer1._style_mean, src_style) * 0.1
    loss = softmax_entropy(outputs).mean(0)

    # loss = loss_mh + loss_ent
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    wandb.log({
        'loss': loss,
        # 'loss/mh': loss_mh,
        # 'loss/ent': loss_ent
    }, commit=False)
    return outputs


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def _old_forward_and_adapt(x: torch.Tensor, model, optimizer, src_style: torch.Tensor, project: nn.Module):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    aug_x = augment(x)
    rc_x = rcrop(x, low=0.7)

    in_x = torch.cat((x, rc_x, aug_x), dim=0)

    outputs, x_emb = model(in_x, return_feature=True)
    gg = F.normalize(project(x_emb), dim=1) #[3B, 1000]

    outputs = torch.split(outputs, x.shape[0])[0]
    gx, gr, gn = torch.split(gg, x.shape[0])

    # _, aug_emb = model(aug_x, return_feature=True)
    # gn = F.normalize(project(aug_emb), dim=1) #[B, 1000]

    # _, rcx_emb = model(rc_x, return_feature=True)
    # gr = F.normalize(project(rcx_emb), dim=1) #[B, 1000]

    # with torch.no_grad():
    #     gs = F.normalize(project(src_style[None, ...]), dim=1) #[1, 1000]

    sim_neg = gx @ gn.T #[B, B]
    sim_pos = torch.bmm(gx.unsqueeze(1), gr.unsqueeze(-1)).reshape(-1, 1) #[B]

    # sim_neg = F.cosine_similarity(gx, gn, dim=1).unsqueeze(-1) #[B]
    # sim_pos = F.cosine_similarity(gx, gs, dim=1).unsqueeze(-1) # [B]
    # loss_cont = (sim_neg - sim_pos).mean(0)

    # loss_pos = (1-F.cosine_similarity(x_emb, crop_emb1, dim=1)).mean(0)
    loss_cont = F.log_softmax(torch.cat((sim_pos, sim_neg), dim=1), dim=1)[:,0] # [B, B+1] -> [B]
    loss_cont = -loss_cont.mean(0)

    # adapt
    loss_ent = softmax_entropy(outputs).mean(0)
    
    loss = loss_cont + loss_ent
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    wandb.log({
        'loss/neg': sim_neg.mean(),
        'loss/pos': sim_pos.mean(),
        'loss/cnt': loss_cont,
        'loss/ent': loss_ent,
        'loss': loss
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