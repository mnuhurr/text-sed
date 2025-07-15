import math
import torch
from .penc import PositionalEncoding
from .utils import SpecAugment, mask_tokens

from dataclasses import dataclass


@dataclass
class CRNNConfig:
    n_classes: int
    dims: list[int]
    kernel_sizes: list[int | tuple[int, int]] | None = None
    poolings: list[int | tuple[int, int] | None] | None = None
    cnn_dropout: float = 0.1
    d_rnn: int = 64
    n_rnn_layers: int = 2
    rnn_dropout: float = 0.1

    n_time_mask: int = 0
    n_freq_mask: int = 0
    time_mask_param: int = 25
    freq_mask_param: int = 8



@dataclass
class CNNTFConfig:
    n_classes: int
    dims: list[int]
    kernel_sizes: list[int | tuple[int, int]] | None = None
    poolings: list[int | tuple[int, int] | None] | None = None
    cnn_dropout: float = 0.1
    d_tf: int = 192
    n_tf_layers: int = 4
    n_heads: int = 8
    tf_dropout: float = 0.1
    p_masking: float = 0.0

    n_time_mask: int = 0
    n_freq_mask: int = 0
    time_mask_param: int = 25
    freq_mask_param: int = 8


class GLU(torch.nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.linear = torch.nn.Conv2d(n_channels, n_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # not exactly the GLU in the literature, but the GLU used in the DESED baseline
        lin = self.linear(x)
        sig = x.sigmoid()
        return lin * sig


class ConvBlock(torch.nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int] = 3,
                 pooling: int | tuple[int, int] | None = None,
                 dropout: float = 0.0):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            torch.nn.BatchNorm2d(out_channels),
            GLU(out_channels),
            torch.nn.Dropout(dropout),
            torch.nn.AvgPool2d(pooling) if pooling is not None else torch.nn.Identity()
        )


class CNN(torch.nn.Module):
    def __init__(self,
                 dims: list[int],
                 kernel_sizes: list[int] | None = None,
                 poolings: list[int | tuple[int, int] | None] | None = None,
                 dropout: float = 0.0,
                 n_time_mask: int = 0,
                 n_freq_mask: int = 0,
                 time_mask_param: int = 50,
                 freq_mask_param: int = 16):
        super().__init__()

        self.specaugment = SpecAugment(
            n_time_mask=n_time_mask,
            n_freq_mask=n_freq_mask,
            time_mask_param=time_mask_param,
            freq_mask_param=freq_mask_param)

        kernel_sizes = kernel_sizes if kernel_sizes is not None else [3] * len(dims)
        poolings = poolings if poolings is not None else [(2, 1)] * len(dims)

        self.blocks = torch.nn.ModuleList()
        dims = [1] + dims

        for k in range(len(dims) - 1):
            block = ConvBlock(
                in_channels=dims[k],
                out_channels=dims[k + 1],
                kernel_size=kernel_sizes[k],
                pooling=poolings[k],
                dropout=dropout)
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.specaugment(x)
        x = x.unsqueeze(1)

        for block in self.blocks:
            x = block(x)

        return x


class CRNN(torch.nn.Module):
    def __init__(self, config: CRNNConfig):
        super().__init__()
        self.config = config

        self.cnn = CNN(
            dims=config.dims,
            kernel_sizes=config.kernel_sizes,
            poolings=config.poolings,
            dropout=config.cnn_dropout)

        self.rnn = torch.nn.GRU(
            batch_first=True,
            bidirectional=True,
            input_size=config.dims[-1],
            hidden_size=config.d_rnn,
            dropout=config.rnn_dropout if config.n_rnn_layers > 1 else 0.0,
            num_layers=config.n_rnn_layers)

        self.head = torch.nn.Linear(2 * config.d_rnn, config.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        
        # x: (batch, ch, f', t'), average out what's left of the frequency dim
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
    
        # x: (batch, t', ch)
        x, _ = self.rnn(x)

        x = self.head(x)
        return x


class CNNTF(torch.nn.Module):
    def __init__(self, config: CNNTFConfig):
        super().__init__()

        self.config = config

        self.cnn = CNN(
            dims=config.dims,
            kernel_sizes=config.kernel_sizes,
            poolings=config.poolings,
            dropout=config.cnn_dropout)

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=config.d_tf,
            dim_feedforward=4 * config.d_tf,
            nhead=config.n_heads,
            dropout=config.tf_dropout,
            batch_first=True,
            norm_first=True)

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(config.dims[-1], config.d_tf),
            torch.nn.LayerNorm(config.d_tf),
            torch.nn.Dropout(config.tf_dropout))

        self.positional_encoding = PositionalEncoding(d_model=config.d_tf, max_sequence_length=1024)
        self.transformer = torch.nn.TransformerEncoder(enc_layer, num_layers=config.n_tf_layers, enable_nested_tensor=False)
        self.register_parameter('mask_token', torch.nn.Parameter(torch.randn(config.d_tf) / math.sqrt(config.d_tf)))

        self.head = torch.nn.Linear(config.d_tf, config.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)

        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)

        x = self.proj(x)

        if self.training and self.config.p_masking > 0:
            x = mask_tokens(x, self.mask_token, p=self.config.p_masking)

        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.head(x)
        return x


