from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple
import torch
from torch.nn import attention
from torch.nn.modules import activation
from wenet.transformer.attention import T_CACHE, MultiHeadedAttention
from wenet.transformer.convolution import ConvolutionModule

from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.utils.class_utils import WENET_ACTIVATION_CLASSES, WENET_MLP_CLASSES, WENET_NORM_CLASSES
from wenet.utils.mask import causal_or_lookahead_mask, make_non_pad_mask


@dataclass
class vocos_config:
    #  convoultion config
    causal = True
    activation = 'gelu'
    conv_norm = 'batch_norm'
    conv_bias = False
    norm_eps = 1e-6
    linear_units = 1536
    kernel_size = 15

    # conformer config
    att_blocks = 3
    input_size = 100
    output_size = 256
    attention_heads = 4
    attention_dropout_rate = 0.1
    qkv_bias = False
    use_sdpa = False
    n_kv_head = 1  # MQA

    # head config
    n_fft = 1024  # 2048
    hop_length = 256  # 640
    padding = 'center'

    dropout_rate = 0.1
    norm_type = 'rms_norm'
    mlp_type = 'position_wise_feed_forward'
    mlp_bias = False
    head_dim = 512


# https://github.com/gemelo-ai/vocos/blob/main/vocos/heads.py#L26
class ISTFTHead(torch.nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, config: vocos_config):
        super().__init__()
        self.dim = config.output_size
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.padding = config.padding
        self.win_length = self.n_fft
        self.window = torch.hann_window(self.win_length)

        out_dim = self.n_fft + 2
        self.out = torch.nn.Linear(self.dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(
            mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        S = mag * (x + 1j * y)
        return torch.istft(S,
                           self.n_fft,
                           self.hop_length,
                           self.win_length,
                           self.window,
                           center=True)


class ConvNoattLayer(ConformerEncoderLayer):

    def __init__(self,
                 size: int,
                 feed_forward: Optional[torch.nn.Module] = None,
                 feed_forward_macaron: Optional[torch.nn.Module] = None,
                 conv_module: Optional[torch.nn.Module] = None,
                 dropout_rate: float = 0.1,
                 normalize_before: bool = True,
                 layer_norm_type: str = 'layer_norm',
                 norm_eps: float = 0.00001):
        super().__init__(size, None, feed_forward, feed_forward_macaron,
                         conv_module, dropout_rate, normalize_before,
                         layer_norm_type, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        assert self.feed_forward is not None
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_cnn_cache


class ConformerConvBeforeAttLayer(ConformerEncoderLayer):

    def __init__(self,
                 size: int,
                 self_attn: torch.nn.Module,
                 feed_forward: Optional[torch.nn.Module] = None,
                 feed_forward_macaron: Optional[torch.nn.Module] = None,
                 conv_module: Optional[torch.nn.Module] = None,
                 dropout_rate: float = 0.1,
                 normalize_before: bool = True,
                 layer_norm_type: str = 'layer_norm',
                 norm_eps: float = 0.00001):
        super().__init__(size, self_attn, feed_forward, feed_forward_macaron,
                         conv_module, dropout_rate, normalize_before,
                         layer_norm_type, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: T_CACHE = (torch.zeros(
            (0, 0, 0, 0)), torch.zeros((0, 0, 0, 0))),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb,
                                              att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        assert self.feed_forward is not None
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


class Vocosv1(torch.nn.Module):
    # modify:
    # two causal convolution  -> three conformer encoder -> head

    def __init__(self, config: vocos_config) -> None:
        super().__init__()

        activation = WENET_ACTIVATION_CLASSES[config.activation]()
        attention_class = MultiHeadedAttention
        attention_layer_args = (
            config.attention_heads,
            config.output_size,
            config.attention_dropout_rate,
            config.qkv_bias,
            config.qkv_bias,
            config.qkv_bias,
            config.use_sdpa,
            config.n_kv_head,
            config.head_dim,
        )
        mlp_class = WENET_MLP_CLASSES[config.mlp_type]
        # feed-forward module definition
        positionwise_layer_args = (
            config.output_size,
            config.linear_units,
            config.dropout_rate,
            activation,
            config.mlp_bias,  # mlp bias
        )

        # convolution module definition
        convolution_layer_args = (
            config.output_size,
            config.kernel_size,
            activation,
            config.conv_norm,
            config.causal,
            config.conv_bias,
        )
        self.linaer = torch.nn.Linear(config.input_size, config.output_size)
        # first two convolution
        self.conv1 = ConvNoattLayer(config.output_size,
                                    mlp_class(*positionwise_layer_args),
                                    mlp_class(*positionwise_layer_args),
                                    ConvolutionModule(*convolution_layer_args),
                                    config.dropout_rate,
                                    layer_norm_type=config.norm_type,
                                    norm_eps=config.norm_eps)
        self.conv2 = ConvNoattLayer(config.output_size,
                                    mlp_class(*positionwise_layer_args),
                                    mlp_class(*positionwise_layer_args),
                                    ConvolutionModule(*convolution_layer_args),
                                    config.dropout_rate,
                                    layer_norm_type=config.norm_type,
                                    norm_eps=config.norm_eps)

        self.encoders = torch.nn.ModuleList([
            ConformerConvBeforeAttLayer(
                config.output_size,
                attention_class(*attention_layer_args),
                mlp_class(*positionwise_layer_args),
                mlp_class(*positionwise_layer_args),
                ConvolutionModule(*convolution_layer_args),
                config.dropout_rate,
                True,  # normalize befor
                'rms_norm',
                config.norm_eps,
            ) for _ in range(config.att_blocks)
        ])

        self.head = ISTFTHead(config)

        self.config = config

    def forward(self, input: torch.Tensor, input_len: torch.Tensor):
        """ forward for training
        """
        x = self.linaer(input)
        mask = make_non_pad_mask(input_len)  # [B,T]

        # TODO: add ln here

        x, _, _ = self.conv1(x, mask.squeeze(1))
        x, _, _ = self.conv2(x, mask.squeeze(1))

        causal_att_mask = causal_or_lookahead_mask(mask.unsqueeze(1), 0, 13)

        # TODO: use sdpa here
        for i, layer in enumerate(self.encoders):
            x, causal_att_mask, _, _ = layer(x, causal_att_mask, None,
                                             mask.unsqueeze(1))

        audio = self.head(x)

        return audio, (mask.sum(1) - 1) * self.config.hop_length


# input = torch.rand(1, 24000)
# input_len = torch.tensor([24000])

# import vocos

# feature = vocos.feature_extractors.MelSpectrogramFeatures()

# config = vocos_config()
# model = Vocosv1(config)
# print(model)
# mels = feature(input)
# print(mels.shape)

# mels = mels.transpose(1, 2)
# gen, gen_lens = model(mels, torch.tensor([mels.shape[1]]))
# print(gen.shape, gen_lens, input.shape)

# print(feature(gen).shape)
