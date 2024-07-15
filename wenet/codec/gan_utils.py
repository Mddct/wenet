import torch
from torch.nn.utils import weight_norm

from wenet.utils.class_utils import WENET_ACTIVATION_CLASSES


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


if __name__ == "__main__":
    model = Discriminator()
    print(sum(p.numel() for p in model.parameters()) / 1_000_000)
    x = torch.randn(1, 128, 1024)
    y, _ = model(x)
    print(y.shape)
