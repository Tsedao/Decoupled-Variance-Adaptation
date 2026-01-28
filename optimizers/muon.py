import torch

## part of code is from modded nonogpt in https://github.com/KellerJordan/modded-nanogpt

def msign(x : torch.Tensor, steps=5, eps=1e-20):
    """ matrix sign function using Newton Schultz iteration
    source https://kexue.fm/archives/10922
    """
    a, b, c, y = 3.4445, -4.7750, 2.0315, x.bfloat16().clone()
    y = y.mT if x.shape[0] > x.shape[1] else y
    ## original code use inplace opertion, which will change the input x 
    ## to the normalized x, so we need to clone the input x
    y /= ((y**2).sum(axis=[-2, -1], keepdims=True) + eps)**0.5
    for i in range(steps):
        y4 = (y2 := y @ y.mT) @ y2
        if i == 0:
            n = ((y4**2).sum(axis=[-2, -1], keepdims=True) + eps)**0.125
            y, y2, y4 = y / n, y2 / n**2, y4 / n**4
        y = a * y + (b * y2 + c * y4) @ y
    return y.mT if x.shape[0] > x.shape[1] else y


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, scaling=1.0):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
        update *= scaling
    update = msign(update, steps=ns_steps)
    update *= max(grad.size(-2), grad.size(-1))**0.5
    return update



class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, nesterov=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
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
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["second_moment_grad_buffer"] = torch.ones(size=(1,)).to(p.device)
                    state["second_moment_momentum_buffer"] = torch.ones(size=(1,)).to(p.device)         
                update = muon_update(p.grad, momentum=state["momentum_buffer"], beta=group["momentum"], nesterov=group["nesterov"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss

