# https://github.com/DequanWang/tent/blob/master/tent.py

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.utils.data

import wandb
import numpy as np
from models.Res import ResNet

def layer_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
    module._style_mean = output.mean((-1, -2)) #style: [B C] <- output: [B C H W]

# Tent with Optimal Transport
class OTent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model: ResNet, optimizer: torch.optim.Optimizer, src_lyrfts: torch.Tensor, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.src_lyrfts = src_lyrfts # [50000, 256], [50000, 512], [50000, 1024], [50000, 2048]
        # self.src_l1ft.requires_grad = False

        self.critics = nn.ModuleList([
            nn.Linear(f.shape[-1], 1) for f in self.src_lyrfts
        ])

        self.optimizers_critic = [
            torch.optim.RMSprop(c.parameters(), lr=5e-5) for c in self.critics
        ]

        # self.critic = nn.Sequential(
        #     nn.Linear(256, 128), #layer1
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 1),
        #     # nn.Sigmoid()
        # )
        # self.optimizer_critic = torch.optim.RMSprop(self.critic.parameters(), lr=5e-5)

        self.model.layer1.register_forward_hook(layer_hook)
        self.model.layer2.register_forward_hook(layer_hook)
        self.model.layer3.register_forward_hook(layer_hook)
        self.model.layer4.register_forward_hook(layer_hook)

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
    
    def features(self):
        return [m._style_mean for m in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]]
    
    def w1_loss(self):
        features = self.features()
        w1s = []

        for crit, ft in zip(self.critics, features):
            w1s.append(-crit(ft).mean(0))

        return torch.cat(w1s).mean()
        
        # return sum(w1s)/len(w1s)


    def forward(self, x: torch.Tensor):
        # if self.episodic:
        #     self.reset()
        # self.train_critic()
        out = self.model(x)

        with torch.enable_grad():
            features = self.features()

            for _ in range(5):
                losses = []
                for critic in self.critics:
                    for p in critic.parameters():
                        p.data.clip_(-0.01, 0.01) #clip to force a Lipschitz constraint to 1

                idx = np.random.choice(len(self.src_lyrfts[0]), 64)
                for srcft, ft, crit, optim in zip(self.src_lyrfts, features, self.critics, self.optimizers_critic):
                    src = srcft[idx].cuda()
                    p_src = crit(src)
                    p_lrn = crit(ft)
                    loss_c = -(p_src.mean() - p_lrn.mean())
                    loss_c.backward()
                    optim.step()
                    optim.zero_grad()

                    losses.append(loss_c)


                # src = self.src_l1ft[idx].cuda()

                # p_src = self.critic(src)
                # p_lrn = self.critic(low_ft)
                # loss_c = -(p_src.mean() - p_lrn.mean())
                # loss_c.backward()

                # self.optimizer_critic.step()
                # self.optimizer_critic.zero_grad()    

            wandb.log({
                'loss/critic': sum(losses) / len(losses)
            }, commit=False)

            for _ in range(self.steps):
                outputs = forward_and_adapt(x, self.model, self.optimizer, self)

        return outputs
    
    # def criticize(self, x: torch.Tensor, src_emb=None):
    #     x_emb: torch.Tensor = self.model(x, return_feature=True, return_feature_only=True)
    #     return criticize(self.src_emb if src_emb is None else src_emb, x_emb, self.critic)
        # p_src: torch.Tensor = self.critic(self.src_emb) # [50000]
        # p_lrn: torch.Tensor = self.critic(embd) # [B]
        # return p_src.mean() - p_lrn.mean()


    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    
    # def train_critic(self):
    #     self.model.requires_grad_(False)
    #     for critic in self.critics:
    #         critic.requires_grad_(True)
    
    # def train_model(self):
    #     configure_model(self.model)
    #     for critic in self.critics:
    #         for p in critic.parameters():
    #             p.requires_grad = False


# def criticize(src_emb, x_emb, critic):
#     p_src: torch.Tensor = critic(src_emb) # [50000]
#     p_lrn: torch.Tensor = critic(x_emb) # [B]
#     return p_src.mean() - p_lrn.mean()

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, otent: OTent):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # otent.train_model()
    # forward
    outputs, x_emb = model(x, return_feature=True)
    # outputs = model(x) 
    # loss_w1 = -otent.critic(x_emb).mean(0)
    # loss_w1 = -otent.critic(model.layer1._style_mean).mean(0)
    loss_w1 = otent.w1_loss()

    # idx = np.random.choice(len(otent.src_emb), 256)
    # src = otent.src_emb[idx].cuda()

    # M = ot.dist(src, x_emb)
    # loss_w1 = ot.emd2(torch.ones(256).cuda()/256, torch.ones(x.shape[0]).cuda()/x.shape[0], M)
    
    # adapt
    loss_ent = softmax_entropy(outputs).mean(0)
    
    loss = loss_w1 # + loss_ent
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    wandb.log({
        'loss/w1': loss_w1,
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