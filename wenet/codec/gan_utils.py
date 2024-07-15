from typing import List, Optional
import torch
from torch.nn.utils import weight_norm
from wenet.transformer.encoder import ConformerEncoder

from wenet.utils.class_utils import WENET_ACTIVATION_CLASSES
from wenet.utils.mask import make_pad_mask


def mel_loss(
    real: torch.Tensor,
    gen: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    splits: List[int] = [],
    weight: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Args:
        real: shape [B, D, T]
        gen: shape [B, D, T]
        mask: shape [B, 1, T]
    """
    mel_l1 = (gen - real).abs()
    splits = [0] + splits + [real.size(1)]
    losses = []
    i = 0
    for (start, end) in zip(splits[:-1], splits[1:]):
        loss = mel_l1[:, start:end, :]
        if weight is not None:
            loss *= weight[i]
            losses.append(loss)
        i += 1
    losses = torch.stack(losses, dim=0)
    if mask is not None:
        losses = (losses * mask) / mask.sum()
    return losses.sum()


class Discriminator(torch.nn.Module):

    def __init__(self,
                 channel=32,
                 activation='swish',
                 activation_params: dict = {}):
        super().__init__()
        # from wenet.utils.class_utils import WENET_ACTIVATION_CLASSES
        activation_class = WENET_ACTIVATION_CLASSES[activation]
        self.convs = torch.nn.Sequential(*[
            weight_norm(torch.nn.Conv2d(1, channel, (3, 9), padding=(1, 4))),
            activation_class(**activation_params),
            weight_norm(
                torch.nn.Conv2d(1 * channel,
                                2 * channel, (3, 9),
                                stride=(1, 2),
                                padding=(1, 4))),
            activation_class(**activation_params),
            weight_norm(
                torch.nn.Conv2d(2 * channel,
                                4 * channel, (3, 9),
                                stride=(1, 2),
                                padding=(1, 4))),
            activation_class(**activation_params),
            weight_norm(
                torch.nn.Conv2d(4 * channel,
                                8 * channel, (3, 9),
                                stride=(1, 2),
                                padding=(1, 4))),
            activation_class(**activation_params),
            weight_norm(
                torch.nn.Conv2d(
                    8 * channel, 16 * channel, (3, 3), padding=(1, 2))),
            activation_class(**activation_params),
            weight_norm(
                torch.nn.Conv2d(16 * channel, 1, (3, 3), padding=(1, 1)))
        ])

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        # x = x.unsqueeze(1)
        intermediate_outputs = []
        for (_, layer) in enumerate(self.convs):
            x = layer(x)
            intermediate_outputs.append(x)

        return x[:, 0], intermediate_outputs


class ConformerDiscriminator(ConformerEncoder):

    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 3,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0,
                 input_layer: str = "linear",
                 pos_enc_layer_type: str = "rel_pos",
                 normalize_before: bool = True,
                 static_chunk_size: int = 0,
                 use_dynamic_chunk: bool = False,
                 global_cmvn: torch.nn.Module = None,
                 use_dynamic_left_chunk: bool = False,
                 positionwise_conv_kernel_size: int = 1,
                 macaron_style: bool = True,
                 selfattention_layer_type: str = "rel_selfattn",
                 activation_type: str = "swish",
                 use_cnn_module: bool = True,
                 cnn_module_kernel: int = 15,
                 causal: bool = False,
                 cnn_module_norm: str = "batch_norm",
                 query_bias: bool = True,
                 key_bias: bool = True,
                 value_bias: bool = True,
                 conv_bias: bool = True,
                 gradient_checkpointing: bool = False,
                 use_sdpa: bool = False,
                 layer_norm_type: str = 'layer_norm',
                 norm_eps: float = 0.00001,
                 n_kv_head: Optional[int] = None,
                 head_dim: Optional[int] = None,
                 mlp_type: str = 'position_wise_feed_forward',
                 mlp_bias: bool = True,
                 n_expert: int = 8,
                 n_expert_activated: int = 2):
        super().__init__(
            input_size, output_size, attention_heads, linear_units, num_blocks,
            dropout_rate, positional_dropout_rate, attention_dropout_rate,
            input_layer, pos_enc_layer_type, normalize_before,
            static_chunk_size, use_dynamic_chunk, global_cmvn,
            use_dynamic_left_chunk, positionwise_conv_kernel_size,
            macaron_style, selfattention_layer_type, activation_type,
            use_cnn_module, cnn_module_kernel, causal, cnn_module_norm,
            query_bias, key_bias, value_bias, conv_bias,
            gradient_checkpointing, use_sdpa, layer_norm_type, norm_eps,
            n_kv_head, head_dim, mlp_type, mlp_bias, n_expert,
            n_expert_activated)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        masks = ~make_pad_mask(x_lens, x.size(1)).unsqueeze(1)  # (B, 1, T)
        x, pos_emb, masks = self.embed(x, masks)

        intermediate_outputs = []
        for (_, layer) in enumerate(self.encoders):
            x, _, _, _ = layer(x, masks, pos_emb)
            intermediate_outputs.append(x)

        return x, intermediate_outputs


if __name__ == "__main__":
    model = Discriminator()
    print(sum(p.numel() for p in model.parameters()) / 1_000_000)
    x = torch.randn(1, 128, 1024)
    y, _ = model(x)
    print(y.shape)

    model = ConformerDiscriminator(input_size=1024, output_size=256)
    print(sum(p.numel() for p in model.parameters()) / 1_000_000)
    x_lens = torch.tensor([128])
    y, _ = model(x, x_lens)
    print(y.shape)
