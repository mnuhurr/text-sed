import math
import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_sequence_length: int = 1024, max_random_offset: int = 0):
        super().__init__()

        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_sequence_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.max_random_offset = max_random_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.max_random_offset > 0:
            offset = torch.randint(self.max_random_offset, size=())
        else:
            offset = 0
        x = x + self.pe[:, offset:offset + x.size(1)]
        return x


class RelativePosConvEncoding(torch.nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()

        pos_conv = torch.nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding='same', groups=d_model)

        # init weight
        std = math.sqrt(4 / (kernel_size * d_model))
        torch.nn.init.normal_(pos_conv.weight, std=std)
        torch.nn.init.constant_(pos_conv.bias, 0.0)

        self.pos_conv = torch.nn.utils.parametrizations.weight_norm(pos_conv, name='weight', dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = x + self.pos_conv(x)
        return x.permute(0, 2, 1)


class RotaryPositionEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, max_sequence_length: int = 1024, n_heads: int = 1, base: int = 10000):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        inv_freq = 1 / (base ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        t = torch.arange(max_sequence_length).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos', torch.cos(freqs))
        self.register_buffer('sin', torch.cos(freqs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x_h = x.reshape(x.size(0), x.size(1), self.n_heads, self.d_head)
        x1, x2 = torch.chunk(x_h, 2, dim=-1)

        cos = self.cos[:seq_len, :].unsqueeze(-2)
        sin = self.sin[:seq_len, :].unsqueeze(-2)

        while cos.dim() < x_h.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        x_h = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return x_h.reshape(x.size(0), x.size(1), self.d_head * self.n_heads)

