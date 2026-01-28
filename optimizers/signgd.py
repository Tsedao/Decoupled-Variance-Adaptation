import torch


class SignSGD(torch.optim.Optimizer):
    def __init__(
            self, 
            params, 
            lr=0.02, 
            weight_decay=0, 
            beta=0.9,
            nestrov= False       
            ):
        defaults = dict(
            lr=lr, 
            weight_decay=weight_decay, 
            beta=beta,
            nestrov=nestrov,
        )
        super().__init__(params, defaults)
        self.global_step = 1

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ## update the momentum
        for group in self.param_groups:
            # torch.nn.utils.clip_grad_norm_(group["params"], 1.0)
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                
                ## decoupled weight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])
                
                ## polyak averaging
                state["momentum_buffer"].lerp_(p.grad, 1-group["beta"])
                update = p.grad.lerp_(state["momentum_buffer"], group["beta"]) if group["nestrov"] else state["momentum_buffer"]

                update = torch.sign(update)
                p.data.add_(update, alpha=-group["lr"])

        self.global_step += 1
        return loss