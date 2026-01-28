import torch
from torch.optim.optimizer import Optimizer

from optimizers.muon import msign

## algorithm 1 from https://arxiv.org/pdf/2507.11005v1

class SingleDeviceAdamuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, betas=(0.95, 0.95), eps=1e-8, nesterov=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["ex_avg"] = torch.zeros_like(p)
                    state["ex_avg_sq"] = torch.zeros_like(p)
                ## first moment
                state["ex_avg"] = group["betas"][0] * state["ex_avg"] + p.grad
                O_t = msign(torch.sign(state["ex_avg"]), steps=5)
                ## second moment
                state["ex_avg_sq"].lerp_((O_t * O_t).float(), 1 - group["betas"][1])
                ## normalize
                O_t = O_t / (state["ex_avg_sq"].sqrt().add_(group["eps"]))
                ## rms alignment
                scaling = 0.2 * ((p.shape[-2] * p.shape[-1]) ** 0.5) / torch.norm(O_t) ## 0.2 /RMS(O_t) in paper
                update = O_t * scaling
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss