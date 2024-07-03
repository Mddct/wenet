import math
from typing import Dict, List, Optional, Tuple, Union
import torch

from wenet.utils.class_utils import WENET_ACTIVATION_CLASSES


def get_extra_padding_for_conv1d(x: torch.Tensor,
                                 kernel_size: int,
                                 stride: int,
                                 padding_total: int = 0) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size -
                                                         padding_total)
    return ideal_length - length


def pad1d(x: torch.Tensor,
          paddings: Tuple[int, int],
          mode: str = 'zero',
          value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left,
                                                      padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = torch.nn.functional.pad(x, (0, extra_pad))
        padded = torch.nn.functional.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return torch.nn.functional.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left,
                                                      padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class SConv1d(torch.nn.Module):
    """ Streaming conv1d eg: causal
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 pad_mode: str = 'zeros') -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    dilation=dilation,
                                    groups=groups,
                                    bias=bias)

        self.pad_mode = pad_mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self, input: torch.Tensor,
                cache: Union[Dict[str, torch.Tensor], None]) -> torch.Tensor:
        """
        Args:
            input: shape (B, channel, T)
        """
        if cache is None:
            kernel_size_with_dilation = (self.kernel_size -
                                         1) * self.dilation + 1
            padding_total = kernel_size_with_dilation - self.stride
            extra_padding = get_extra_padding_for_conv1d(
                input, kernel_size_with_dilation, self.stride, padding_total)
            input = pad1d(input, (padding_total, extra_padding),
                          mode=self.pad_mode)
            return self.conv(input)
        else:
            conv_cache = cache['conv']
            input = torch.cat((conv_cache, input), dim=-1)
            cache['conv'].copy_(input[:, self.stride:, ])
            output = self.conv(input)
            return output


class SConvTransposed1d(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 pad_mode: str = 'zeros') -> None:
        super().__init__()
        self.convT = torch.nn.ConvTranspose1d(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride,
                                              groups=groups,
                                              bias=bias)
        self.pad_mode = pad_mode
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: torch.Tensor,
                cache: Union[Dict[str, torch.Tensor], None]) -> torch.Tensor:
        padding_total = self.kernel_size - self.stride
        if cache is None:
            x = self.convT(input)
            padding_left = 0
            padding_right = padding_total
            x = unpad1d(x, (padding_left, padding_right))
            return x

        else:
            raise NotImplementedError('cache is not supported for now')


class SResidualUnit(torch.nn.Module):
    """Streaming residual unit
    """

    def __init__(self,
                 in_channels: int,
                 hidden: int,
                 kernel_sizes: List[int],
                 dilations: List[int],
                 activation_type: str = 'elu',
                 activation_params: Dict = {},
                 scale: float = 1.0,
                 shortcut: bool = False) -> None:
        super().__init__()
        assert len(kernel_sizes) == len(dilations)

        activation_class = WENET_ACTIVATION_CLASSES[activation_type]
        self.activation0 = activation_class(**activation_params)
        self.activation1 = activation_class(**activation_params)
        assert len(kernel_sizes) == 2
        assert len(dilations) == 2
        self.conv0 = SConv1d(
            in_channels,
            hidden,
            kernel_size=kernel_sizes[0],
            dilation=dilations[0],
            pad_mode='constant',  # not reflect for streaming
            bias=True,
            groups=hidden,
        )  # pointwise
        self.conv1 = SConv1d(
            hidden,
            hidden,
            kernel_size=kernel_sizes[1],
            dilation=dilations[1],
            pad_mode='constant',  # not reflect for streaming
            bias=True,
            groups=1,
        )  # depthwise
        self.residual_scalar = scale
        self.skip = shortcut

    def forward(
            self,
            input: torch.Tensor,
            cache: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        x = self.activation0(input)
        x = self.conv0(x, cache)
        x = self.activation1(x)
        x = self.conv1(x, cache)
        if self.skip:
            return x + self.residual_scalar * input
        else:
            return x


class EncoderBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        residual_kernel_width: int = 3,
        stride: int = 2,
        num_residual_layers=3,
        activation: str = 'elu',
        activation_params: dict = {},
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.residual_kernel_width = residual_kernel_width
        self.stride = stride
        self.num_residual_layers = num_residual_layers
        compress_channels = hidden // 2
        self.blocks = torch.nn.ModuleList([
            SResidualUnit(in_channels if idx == 0 else compress_channels,
                          compress_channels,
                          kernel_sizes=[residual_kernel_width, 1],
                          dilations=[residual_kernel_width**idx, 1],
                          activation_type=activation,
                          activation_params=activation_params,
                          shortcut=(idx != 0))
            for idx in range(self.num_residual_layers)
        ])
        self.norm = torch.nn.LayerNorm(compress_channels, eps=norm_eps)
        self.activation = WENET_ACTIVATION_CLASSES[activation](
            activation_params)
        self.final_conv = SConv1d(
            compress_channels,
            hidden,
            2 * stride,
        )

    def forward(self, input: torch.Tensor,
                cache: Union[Dict[str, torch.Tensor], None]) -> torch.Tensor:
        # TODO(Mddct): fix this cache later
        x = input
        for layer in self.blocks:
            x = layer(x, None)
        x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        x = x.transpose(1, 2)
        return self.final_conv(x, cache)


class SeaEncoder(torch.nn.Module):

    def __init__(
        self,
        hidden: int,
        bottle_hidden: int,
        ratios: List[int],
        num_residual_layers=3,
        kernel_size: int = 7,
        bottle_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.conv0 = SConv1d(1,
                             hidden,
                             kernel_size=kernel_size,
                             pad_mode='constant')
        self.norm0 = torch.nn.LayerNorm(hidden, eps=1e-6)
        self.act = WENET_ACTIVATION_CLASSES['swish']()
        in_channels = hidden
        blocks = []
        mult = 1

        out_channels = hidden
        for ratio in ratios:
            in_channels = out_channels
            out_channels = hidden * mult
            blocks.append(
                EncoderBlock(
                    in_channels=in_channels,
                    hidden=out_channels,
                    num_residual_layers=num_residual_layers,
                    stride=ratio,
                    activation='swish',
                ))
            mult *= 2

        self.encoders = torch.nn.ModuleList(blocks)
        # Bottleneck
        self.bottleneck = SConv1d(out_channels,
                                  bottle_hidden,
                                  kernel_size=bottle_kernel_size,
                                  pad_mode='constant')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: shape [B,T,1]
        """
        input = input.transpose(1, 2)
        x = self.conv0(input, None)
        x = self.act(self.norm0(x.transpose(1, 2)))
        x = x.transpose(1, 2)

        # TODO: fix cache
        for _, layer in enumerate(self.encoders):
            x = layer(x, None)
        x = self.bottleneck(x, None)
        return x


if __name__ == '__main__':
    encoder = SeaEncoder(128, 128, [2, 4, 5, 8])
    print(sum(p.numel() for p in encoder.parameters()))
