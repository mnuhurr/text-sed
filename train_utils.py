import numpy as np
import torch


@torch.no_grad()
def spectrogram_mixing(x: torch.Tensor, max_amount: float = 0.3) -> torch.Tensor:
    u = max_amount * torch.rand(x.size(0), 1, 1, device=x.device)
    return (1 - u) * x + u * torch.flip(x, dims=(0,))


@torch.no_grad()
def batch_mixup(x: torch.Tensor, y: torch.Tensor, mixup_alpha: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
    if x.size(0) % 2 == 1:
        x = x[:-1]
        y = y[:-1]

    t = np.random.beta(mixup_alpha, mixup_alpha)
    n = x.size(0) // 2

    x_u = t * x[:n] + (1 - t) * x[n:]
    x_d = t * x[n:] + (1 - t) * x[:n]

    y_u = t * y[:n] + (1 - t) * y[n:]
    y_d = t * y[n:] + (1 - t) * y[:n]

    x_mu = torch.cat([x_u, x_d], dim=0)
    y_mu = torch.cat([y_u, y_d], dim=0)
    return x_mu, y_mu


@torch.no_grad()
def batch_pos_weight(y_true: torch.Tensor) -> torch.Tensor:
    tags, _ = torch.max(y_true, dim=1)
    p_pos = torch.mean(tags, dim=0)
    w = torch.ones_like(p_pos)
    idx = p_pos > 0
    w[idx] = (1 - p_pos[idx]) / p_pos[idx]
    return w


def entropy(x):
    probs = x / torch.sum(x)
    ind = probs > 0
    return -torch.sum(probs[ind] * torch.log2(probs[ind]))


def optimizer_size(optimizer: torch.optim.Optimizer) -> int:
    tot_bytes = 0

    for g in optimizer.param_groups:
        for p in g['params']:
            for v in optimizer.state[p].values():
                if isinstance(v, torch.Tensor):
                    tot_bytes += v.numel() * v.element_size()

    return tot_bytes
