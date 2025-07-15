import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from functools import reduce

from .crnn import CNN
from .penc import PositionalEncoding
from .utils import SpecAugment
from .utils import mask_tokens
from .utils import AttentionAveragePooling


@dataclass
class AudioEncoderConfig:
    n_mels: int
    d_model: int
    n_heads: int = 8
    n_layers: int = 6
    n_enc_channels: int = 128
    #enc_stack_size: int = 4
    n_enc_blocks: int = 2
    p_masking: float = 0.0
    dropout: float = 0.0


@dataclass
class TextEncoderConfig:
    vocab_size: int
    d_model: int
    n_heads: int = 8
    n_layers: int = 6
    p_masking: float = 0.0
    mask_token_id: int = 4
    dropout: float = 0.0


@dataclass
class MultimodalSEDConfig:
    n_classes: int
    n_mels: int
    vocab_size: int
    d_model: int
    n_audio_enc_heads: int = 8
    n_audio_enc_layers: int = 2
    n_audio_enc_cnn_channels: int = 256
    #audio_enc_stack_size: int = 4
    n_audio_enc_blocks: int = 2
    p_audio_enc_masking: float = 0.0
    n_text_enc_heads: int = 8
    n_text_enc_layers: int = 6
    p_text_enc_masking: float = 0.0
    mask_token_id: int = 4
    n_dec_layers: int = 4
    n_dec_heads: int = 8
    dropout: float = 0.0


@dataclass
class CNNMultimodalSEDConfig:
    n_classes: int
    vocab_size: int

    # cnn
    dims: list[int]
    kernel_sizes: list[int] | None = None
    poolings: list[int | tuple[int, int] | None] | None = None
    cnn_dropout: float = 0.0
    n_time_mask: int = 0
    n_freq_mask: int = 0
    time_mask_param: int = 50
    freq_mask_param: int = 16

    # common for decoder and text encoder
    d_tf: int = 192
    p_text_masking: float = 0.0

    # text encoder
    n_text_enc_heads: int = 8
    n_text_enc_layers: int = 6
    mask_token_id: int = 4
    text_dropout: float = 0.0

    # multimodal decoder/fusing
    n_dec_layers: int = 4
    n_dec_heads: int = 8
    dec_dropout: float = 0.0
    p_decoder_masking: float = 0.0

    # for weak labels ('mean', 'attn')
    weak_pooling: str = 'mean'


@dataclass
class MultimodalCRNNConfig:
    n_classes: int
    vocab_size: int
    dims: list[int]
    kernel_sizes: list[int | tuple[int, int]] | None = None
    poolings: list[int | tuple[int, int] | None] | None = None
    cnn_dropout: float = 0.1
    d_rnn: int = 64
    n_rnn_layers: int = 2
    rnn_dropout: float = 0.1

    # text tf
    d_tf: int = 128
    n_tf_layers: int = 4
    n_tf_heads: int = 8
    tf_dropout: float = 0.0
    p_text_masking: float = 0.0
    mask_token_id = 4

    ca_dropout: float = 0.2

    n_time_mask: int = 0
    n_freq_mask: int = 0
    time_mask_param: int = 25
    freq_mask_param: int = 8

    # for weak labels ('mean', 'attn')
    weak_pooling: str = 'mean'


def _init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)


class TransformerEncoderBackbone(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 p_masking: float = 0.0,
                 dropout: float = 0.0,
                 max_tokens: int = 1024,
                 positional_encoding: torch.nn.Module | None = None):
        super().__init__()

        self.p_masking = p_masking
        self.positional_encoding = positional_encoding if positional_encoding is not None else PositionalEncoding(d_model=d_model, max_sequence_length=max_tokens)

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=4 * d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True)

        self.transformer = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)

        self.register_parameter('mask_token', torch.nn.Parameter(torch.randn(d_model) / math.sqrt(d_model)))

        self.apply(_init_weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.training and self.p_masking > 0:
            x = mask_tokens(x, self.mask_token, p=self.p_masking)

        x = self.positional_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)

        return x


class AudioEncoder(torch.nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config

        self.encoder = CNN1dEncoder(
            n_channels=config.n_enc_channels,
            #stack_size=config.enc_stack_size,
            n_blocks=config.n_enc_blocks,
            d_model=config.d_model,
            n_mels=config.n_mels,
            dropout=config.dropout)

        self.transformer = TransformerEncoderBackbone(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            p_masking=config.p_masking,
            dropout=config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.encoder(x)
        
        # if the mask is provided, we need to squeeze down it by the factor of enc_stack_size
        if mask is not None and self.config.enc_stack_size > 1:
            mask = mask.float()
            mask = F.avg_pool1d(mask, self.config.enc_stack_size)
            # take floor: any position that contains a bit of audio are taken into account
            # take ceil: all the positions in a frame must contain input
            mask = torch.floor(mask).bool()

        x = self.transformer(x, mask=mask)
        return x, mask


class DecoderBlock(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        self.ln0 = torch.nn.LayerNorm(d_model)
        self.mha0 = torch.nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.ln1 = torch.nn.LayerNorm(d_model)
        self.mha1 = torch.nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.ln2 = torch.nn.LayerNorm(d_model)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.GELU(),
            torch.nn.Linear(4 * d_model, d_model),
            torch.nn.Dropout(dropout))

    def forward(self,
                x: torch.Tensor,
                xa: torch.Tensor | None = None,
                x_mask: torch.Tensor | None = None,
                xa_mask: torch.Tensor | None = None) -> torch.Tensor:
        # 1. self attention
        x = self.ln0(x)
        x_attn, sa_scores = self.mha0(x, x, x, key_padding_mask=x_mask)
        x = x + x_attn

        xa_scores = None
        if xa is not None:
            # skip cross attention for the batch items that are completely masked out
            idx = torch.ones(x.size(0), dtype=bool)
            if xa_mask is not None:
                idx = torch.logical_not(torch.all(xa_mask, dim=1))

            x[idx] = self.ln1(x[idx])
            x_attn, xa_scores = self.mha1(x[idx], xa[idx], xa[idx], key_padding_mask=xa_mask[idx] if xa_mask is not None else None)
            x[idx] = x[idx] + x_attn

        x = self.ln2(x)
        x = x + self.fc(x)

        return x


class CNN1dEncoder(torch.nn.Module):
    def __init__(self, n_channels: int, d_model: int, n_mels: int, dropout: float = 0.1, n_blocks: int = 2):
        super().__init__()
        
        blocks = []
        in_channels = n_mels
        out_channels = n_channels
        for _ in range(n_blocks):
            blocks.append(
                torch.nn.Sequential(
                    torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same'),
                    torch.nn.BatchNorm1d(out_channels),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.AvgPool1d(2),
                )
            )
            in_channels = out_channels

        self.cnn = torch.nn.Sequential(*blocks)

        self.head = torch.nn.Conv1d(n_channels, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.head(x)
        x = x.permute(0, 2, 1)
        return x


class CNNEncoder(torch.nn.Module):
    def __init__(self, n_channels: int, d_model: int, n_mels: int, dropout: float = 0.1, n_blocks: int = 2):
        super().__init__()

        self.specaugment = SpecAugment(
            n_time_mask=2,
            n_freq_mask=2,
            time_mask_param=50,
            freq_mask_param=16)

        self.cnn0 = torch.nn.Conv2d(1, n_channels, kernel_size=3, padding='same')

        self.blocks = torch.nn.Sequential(*[
            torch.nn.Sequential(
                torch.nn.Conv2d(n_channels, n_channels, kernel_size=3, padding='same'),
                torch.nn.BatchNorm2d(n_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),

                torch.nn.Conv2d(n_channels, n_channels, kernel_size=3, padding='same'),
                torch.nn.BatchNorm2d(n_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),

                torch.nn.AvgPool2d((2, 1)),
            ) for _ in range(n_blocks)
        ])

        d_out = n_channels * (n_mels // (2**n_blocks))
        self.head = torch.nn.Linear(d_out, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.specaugment(x)
        x = self.cnn0(x.unsqueeze(1))
        x = self.blocks(x)
        # x: (batch, ch, m', t)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.head(x)
        return x


class TextEncoder(torch.nn.Module):
    def __init__(self, config: TextEncoderConfig):
        super().__init__()

        self.config = config

        self.embedding = torch.nn.Embedding(config.vocab_size, config.d_model)

        self.transformer = TransformerEncoderBackbone(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            p_masking=0,
            dropout=config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.training and self.config.p_masking > 0:
            idx = torch.rand(x.shape, device=x.device) < self.config.p_masking
            x[idx & (x > self.config.mask_token_id)] = self.config.mask_token_id

        x = self.embedding(x)
        x = self.transformer(x)
        return x


class MultimodalDecoder(torch.nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        self.blocks = torch.nn.ModuleList([
            DecoderBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])

    def forward(self,
                audio: torch.Tensor,
                text: torch.Tensor | None = None,
                audio_mask: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None):

        x = audio
        for block in self.blocks:
            x = block(x=x, xa=text, x_mask=audio_mask, xa_mask=text_mask)

        return x


class MultimodalSED(torch.nn.Module):
    def __init__(self, config: MultimodalSEDConfig):
        super().__init__()

        self.config = config

        audio_cfg = AudioEncoderConfig(
            n_mels=config.n_mels,
            d_model=config.d_model,
            n_heads=config.n_audio_enc_heads,
            n_layers=config.n_audio_enc_layers,
            n_enc_channels=config.n_audio_enc_cnn_channels,
            p_masking=config.p_audio_enc_masking,
            #enc_stack_size=config.audio_enc_stack_size,
            n_enc_blocks=config.n_audio_enc_blocks,
            dropout=config.dropout)

        text_cfg = TextEncoderConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_text_enc_heads,
            n_layers=config.n_text_enc_layers,
            p_masking=config.p_text_enc_masking,
            mask_token_id=config.mask_token_id,
            dropout=config.dropout)

        self.audio_encoder = AudioEncoder(audio_cfg)
        self.text_encoder = TextEncoder(text_cfg)

        self.decoder = MultimodalDecoder(
            d_model=config.d_model,
            n_layers=config.n_dec_layers,
            n_heads=config.n_dec_heads,
            dropout=config.dropout)

        self.head = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, 2 * config.d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(2 * config.d_model, config.n_classes))

    def forward(self,
                audio: torch.Tensor,
                text: torch.Tensor | None = None,
                audio_mask: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:

        xa, audio_mask = self.audio_encoder(audio, mask=audio_mask)
        xt = self.text_encoder(text, mask=text_mask) if text is not None else None
        y = self.decoder(audio=xa, text=xt, audio_mask=audio_mask, text_mask=text_mask)

        y = self.head(y)
        return y


class CNNMultimodalSED(torch.nn.Module):
    def __init__(self, config: CNNMultimodalSEDConfig):
        super().__init__()

        self.config = config

        time_factors = [p for _, p in config.poolings]
        self.downsampling_factor = reduce(lambda x, y: x * y, time_factors, 1)

        self.cnn = CNN(
            dims=config.dims,
            kernel_sizes=config.kernel_sizes,
            poolings=config.poolings,
            dropout=config.cnn_dropout,
            n_time_mask=config.n_time_mask,
            n_freq_mask=config.n_freq_mask,
            time_mask_param=config.time_mask_param,
            freq_mask_param=config.freq_mask_param)

        text_cfg = TextEncoderConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_tf,
            n_heads=config.n_text_enc_heads,
            n_layers=config.n_text_enc_layers,
            p_masking=config.p_text_masking,
            mask_token_id=config.mask_token_id,
            dropout=config.text_dropout)

        self.text_encoder = TextEncoder(text_cfg)

        self.audio_norm = torch.nn.LayerNorm(config.dims[-1])
        self.text_norm = torch.nn.LayerNorm(config.d_tf)
        self.ca = CrossAttention(config.dims[-1], config.d_tf, n_heads=1, dropout=config.dec_dropout)
        self.ca_norm = torch.nn.LayerNorm(config.dims[-1])
        self.audio_proj = torch.nn.Linear(config.dims[-1], config.d_tf)

        # we need positional encoding for the audio tokens
        #self.positional_encoding = PositionalEncoding(d_model=config.d_tf, max_sequence_length=1024)

        self.decoder = TransformerEncoderBackbone(
            d_model=config.d_tf,
            n_layers=config.n_dec_layers,
            n_heads=config.n_dec_heads,
            dropout=config.dec_dropout,
            positional_encoding=torch.nn.Identity(),
            p_masking=config.p_decoder_masking)
        """
        self.decoder = MultimodalDecoder(
            d_model=config.d_tf,
            n_layers=config.n_dec_layers,
            n_heads=config.n_dec_heads,
            dropout=config.dropout)
        """

        assert config.weak_pooling in ['attn', 'mean', 'panns', 'adaptive-panns']
        if config.weak_pooling == 'attn':
            self.weak_scores = AttentionAveragePooling(config.d_tf, config.n_classes)
        elif config.weak_pooling == 'adaptive-panns':
            self.register_parameter('pooling_weights', torch.nn.Parameter(torch.tensor([1.0, 1.0])))
        else:
            self.weak_scores = None

        self.head = torch.nn.Linear(config.d_tf, config.n_classes)

        #self.register_parameter('mask_token', torch.nn.Parameter(torch.randn(config.d_tf) / math.sqrt(config.d_tf)))
        self.apply(_init_weight)

    def _contract_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        mask = F.avg_pool1d(mask, self.downsampling_factor)
        # take floor: any position that contains a bit of audio are taken into account
        # take ceil: all the positions in a frame must contain input
        mask = torch.floor(mask).bool()
        return mask

    def _handle_text_input(self, x: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        if text_mask is not None:
            idx = torch.logical_not(torch.all(text_mask, dim=1))
            if not torch.any(idx):
                return x
        else:
            idx = torch.ones(x.size(0), dtype=bool, device=text.device)

        y = self.text_encoder(text[idx], mask=text_mask[idx])
        y = self.text_norm(y)

        y_ca = self.ca(x[idx], y)

        x[idx] = x[idx] + y_ca  # self.ca(x[idx], y)
        x[idx] = self.ca_norm(x[idx])
        return x

    def forward(self,
                audio: torch.Tensor,
                text: torch.Tensor | None = None,
                audio_mask: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None,
                return_weak: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        xa = self.cnn(audio)
        xa = xa.mean(dim=2)
        xa = xa.permute(0, 2, 1)
        x = self.audio_norm(xa)

        audio_mask = self._contract_mask(audio_mask) if audio_mask is not None else None

        # x: (batch, t', ch)

        if text is not None:
            x = self._handle_text_input(x, text, text_mask)

        x = self.audio_proj(x)
        x = self.decoder(x)

        """
        xt = self.text_encoder(text, mask=text_mask) if text is not None else None

        # corrupt audio tokens if we are training
        if self.training and self.config.p_audio_masking > 0:
            xa = mask_tokens(xa, self.mask_token, p=self.config.p_audio_masking)
        xa = self.positional_encoding(xa)

        # this handles the pos encoding and masking
        y = self.decoder(audio=xa, text=xt, audio_mask=audio_mask, text_mask=text_mask)
        """

        x_strong = self.head(x)

        if not return_weak:
            return x_strong

        if self.weak_scores is not None:
            x_weak = self.weak_scores(x, x_strong)
        else:
            if self.config.weak_pooling == 'mean':
                x_weak = torch.mean(x_strong, dim=1)
            elif self.config.weak_pooling == 'panns' or self.config.weak_pooling == 'adaptive-panns':
                x1 = torch.mean(x_strong, dim=1)
                x2, _ = torch.max(x_strong, dim=1)

                if self.config.weak_pooling == 'panns':
                    x_weak = 0.5 * (x1 + x2)
                else:
                    w = self.pooling_weights.tanh()
                    x_weak = w[0] * x1 + w[1] * x2

        return x_strong, x_weak


class SEDTransformer(torch.nn.Module):
    def __init__(self, config: AudioEncoderConfig, n_labels: int):
        super().__init__()

        self.config = config
        self.encoder = AudioEncoder(config)
        self.post_norm = torch.nn.LayerNorm(config.d_model)
        self.post_dropout = torch.nn.Dropout(config.dropout)
        self.head = torch.nn.Linear(config.d_model, n_labels)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.encoder(x, mask=x_mask)
        x = self.post_norm(x)
        x = self.post_dropout(x)
        x = self.head(x)
        return x


class CrossAttention(torch.nn.Module):
    def __init__(self, d_audio: int, d_text: int, n_heads: int = 1, dropout: float = 0.0):
        super().__init__()

        #self.text_proj = torch.nn.Linear(d_text, d_audio)
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=d_audio,
            kdim=d_text,
            vdim=d_text,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True)

    def forward(self, audio: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        #text = self.text_proj(text)
        xa, _ = self.mha(audio, text, text, key_padding_mask=text_mask)
        return xa


class MultimodalCRNN(torch.nn.Module):
    def __init__(self, config: MultimodalCRNNConfig):
        super().__init__()
        self.config = config

        self.cnn = CNN(
            dims=config.dims,
            kernel_sizes=config.kernel_sizes,
            poolings=config.poolings,
            dropout=config.cnn_dropout)

        enc_cfg = TextEncoderConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_tf,
            n_layers=config.n_tf_layers,
            n_heads=config.n_tf_heads,
            dropout=config.tf_dropout,
            p_masking=config.p_text_masking,
            mask_token_id=config.mask_token_id)

        self.text_encoder = TextEncoder(enc_cfg)

        self.audio_norm = torch.nn.LayerNorm(config.dims[-1])
        self.text_norm = torch.nn.LayerNorm(config.d_tf)
        self.ca = CrossAttention(config.dims[-1], config.d_tf, n_heads=config.n_tf_heads, dropout=config.tf_dropout)
        self.ca_norm = torch.nn.LayerNorm(config.dims[-1])

        self.rnn = torch.nn.GRU(
            batch_first=True,
            bidirectional=True,
            input_size=2 * config.dims[-1],
            hidden_size=config.d_rnn,
            dropout=config.rnn_dropout if config.n_rnn_layers > 1 else 0.0,
            num_layers=config.n_rnn_layers)

        assert config.weak_pooling in ['attn', 'mean', 'panns', 'adaptive-panns']
        if config.weak_pooling == 'attn':
            #self.weak_scores = torch.nn.Linear(2 * config.d_rnn, config.n_classes)
            self.weak_scores = AttentionAveragePooling(2 * config.d_rnn, config.n_classes)
        elif config.weak_pooling == 'adaptive-panns':
            self.register_parameter('pooling_weights', torch.nn.Parameter(torch.tensor([1.0, 1.0])))
            self.weak_scores = None
        else:
            self.weak_scores = None

        self.head = torch.nn.Linear(2 * config.d_rnn, config.n_classes)

    def _handle_text_input(self, x: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        if text is None:
            xt = torch.zeros_like(x)
            return torch.cat([x, xt], dim=-1)

        if text_mask is not None:
            idx = torch.logical_not(torch.all(text_mask, dim=1))
            if not torch.any(idx):
                return x
        else:
            idx = torch.ones(x.size(0), dtype=bool, device=text.device)

        y = self.text_encoder(text[idx], mask=text_mask[idx])
        y = self.text_norm(y)

        ca = self.ca(x[idx], y)
        y_ca = torch.zeros_like(x, dtype=ca.dtype)
        y_ca[idx] = self.ca(x[idx], y)
        """
        if self.training and self.config.ca_dropout > 0:
            dropout_idx = torch.rand(y_ca.size(0), y_ca.size(1)) < 0.2
            y_ca[dropout_idx] = 0
        """
        #x[idx] = x[idx] + y_ca  # self.ca(x[idx], y)
        #x[idx] = self.ca_norm(x[idx])
        x = torch.cat([x, y_ca], dim=-1)
        return x

    def forward(self, audio: torch.Tensor,
                text: torch.Tensor | None = None,
                audio_mask: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None,
                return_weak: bool = False,
                return_features: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        x = self.cnn(audio)

        # x: (batch, ch, f', t'), average out what's left of the frequency dim
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        x = self.audio_norm(x)

        #if text is not None:
        x = self._handle_text_input(x, text, text_mask)

        # x: (batch, t', ch)
        x, _ = self.rnn(x)

        x_strong = self.head(x)

        if not return_weak:
            if return_features:
                return x_strong, x
            return x_strong

        if self.weak_scores is not None:
            #s = self.weak_scores(x)
            #s = torch.softmax(s, dim=-1)
            #s = torch.clamp(s, min=1e-7, max=1)
            #x_weak = torch.sum(s * x_strong, dim=1) / torch.sum(s, dim=1)
            x_weak = self.weak_scores(x, x_strong)
        else:
            if self.config.weak_pooling == 'mean':
                x_weak = torch.mean(x_strong, dim=1)
            elif self.config.weak_pooling == 'panns' or self.config.weak_pooling == 'adaptive-panns':
                x1 = torch.mean(x_strong, dim=1)
                x2, _ = torch.max(x_strong, dim=1)

                if self.config.weak_pooling == 'panns':
                    x_weak = 0.5 * (x1 + x2)
                else:
                    w = self.pooling_weights.tanh()
                    x_weak = w[0] * x1 + w[1] * x2

        if return_features:
            return x_strong, x_weak, x

        return x_strong, x_weak

