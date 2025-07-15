import torch
import torchaudio
from collections import OrderedDict


def model_size(model: torch.nn.Module) -> int:
    return sum([p.numel() for p in model.parameters()])


@torch.jit.script
def mask_tokens(x: torch.Tensor, mask_token: torch.Tensor, p: float) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape

    mask_pos = torch.rand(batch_size, seq_len, device=x.device) < p
    x[mask_pos] = mask_token.to(x.dtype)

    return x


@torch.jit.script
def topk_latent(z: torch.Tensor, k: int) -> torch.Tensor:
    v, idx = torch.topk(z, k)

    zk = torch.zeros_like(z)
    zk.scatter_(-1, idx, v)

    return zk


@torch.no_grad()
def ema_update(online: torch.nn.Module, target: torch.nn.Module, decay: float = 0.999):
    """
    ema update target model weights from online model using given decay.

    :param online: online model
    :param target: model to update
    :param decay: decay for ema
    :return: None
    """

    # 1. update parameters
    model_params = OrderedDict(online.named_parameters())
    target_params = OrderedDict(target.named_parameters())

    assert model_params.keys() == target_params.keys()

    for name, param in model_params.items():
        target_params[name].sub_((1.0 - decay) * (target_params[name] - param))

    # 2. copy buffers
    model_buffers = OrderedDict(online.named_buffers())
    target_buffers = OrderedDict(target.named_buffers())

    assert model_buffers.keys() == target_buffers.keys()

    for name, buffer in model_buffers.items():
        target_buffers[name].copy_(buffer)


class MelScaling(torch.nn.Module):
    def __init__(self, n_mels: int, mu: float = 0.0, sigma: float = 1.0):
        super().__init__()
        log_var = 2 * torch.log(torch.as_tensor(sigma))
        self.mean = torch.nn.Parameter(mu * torch.ones(1, n_mels, 1))
        self.log_var = torch.nn.Parameter(log_var * torch.ones(1, n_mels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * self.log_var)
        return (x - self.mean) / std


class SpecAugment(torch.nn.Module):
    def __init__(self,
                 n_time_mask: int,
                 n_freq_mask: int,
                 time_mask_param: int = 50,
                 freq_mask_param: int = 16):
        super().__init__()

        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)

        self.n_time_mask = n_time_mask
        self.n_freq_mask = n_freq_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        for _ in range(self.n_time_mask):
            x = self.time_mask(x)

        for _ in range(self.n_freq_mask):
            x = self.freq_mask(x)

        return x


class DropPath(torch.nn.Module):
    """stochastic depth/drop path"""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()

        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device).bernoulli_(keep_prob)
        if self.scale_by_keep:
            mask.div_(keep_prob)

        return mask * x


class AttentionAveragePooling(torch.nn.Module):
    def __init__(self, d_features: int, num_classes: int, dropout: float = 0.0, use_norm: bool = True):
        super().__init__()

        self.norm = torch.nn.LayerNorm(d_features) if use_norm else torch.nn.Identity()
        self.qk = torch.nn.Linear(d_features, 2 * d_features, bias=False)
        self.dropout = dropout

    def forward(self, features: torch.Tensor, x_strong: torch.Tensor) -> torch.Tensor:
        d = features.size(-1)
        x = self.norm(features)
        q, k = self.qk(x).split(d, dim=-1)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=x_strong,
            dropout_p=self.dropout if self.training else 0.0)

        return attn_out.mean(dim=1)

def _test():
    z = torch.randn(4, 12, 64)
    x = torch.randn(4, 12, 17)
    pool = AttentionAveragePooling(64, 17)
    y = pool(z, x)
    print(y.shape)

if __name__ == '__main__':
    _test()

