import math
from typing import Dict, List, Tuple, Union
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
                 pad_mode: str = 'reflect') -> None:
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
                cache: Union[Dict[str, torch.Tensor], None]):
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


class SResidualUnit(torch.nn.Module):
    """Streaming residual unit
    """

    def __init__(self,
                 in_channels: int,
                 hidden: int,
                 out_channels: int,
                 kernel_sizes: List[int],
                 dilations: List[int],
                 activation_type: str = 'elu',
                 activation_params: Dict = {},
                 scale: float = 1.0) -> None:
        super().__init__()
        assert len(kernel_sizes) == len(dilations)

        activation_class = WENET_ACTIVATION_CLASSES[activation_type]
        assert len(kernel_sizes) == 2
        assert len(dilations) == 2
        self.conv0 = SConv1d(
            in_channels,
            hidden,
            kernel_size=kernel_sizes[0],
            dilation=dilations[0],
            pad_mode='constant',  # not reflect for streaming
            bias=True,
            groups=in_channels,
        )  # pointwise
        self.norm0 = activation_class(**activation_params)
        self.conv1 = SConv1d(
            hidden,
            out_channels,
            kernel_size=kernel_sizes[1],
            dilation=dilations[1],
            pad_mode='constant',  # not reflect for streaming
            bias=True,
            groups=1,
        )  # depthwise
        self.norm1 = activation_class(**activation_params)
        self.residual_scalar = scale

    def forward(self, input: torch.Tensor,
                cache: Union[Dict[str, torch.Tensor], None]):
        x = input
        x = self.conv0(input, cache)

        x = x.transpose(1, 2)
        x = self.norm0(x)
        x = x.transpose(1, 2)
        x = self.conv1(x, cache)

        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)

        return x + self.residual_scalar * input
