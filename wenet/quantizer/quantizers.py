from dataclasses import dataclass
from typing import List, Optional

import torch
import enum


@dataclass
class QuantizerOutputs:
    quantized: torch.Tensor
    quantization_loss: torch.Tensor
    nn_idx: torch.Tensor
    codebook: torch.Tensor
    cluster_counts: List[torch.Tensor]


class QuantizationStrategy(enum.Enum):
    """The Quantization strategy."""

    PRODUCT_QUANTIZATION = 'product_quantization'
    RESIDUAL_QUANTIZATION = 'residual_quantization'


class BaseQuantizer(torch.nn.Module):
    """Quantizer that can be used as a building block for ProductQuantizer."""

    def __init__(self,
                 num_centroids,
                 stop_gradient_codes=True,
                 ema_decay=0.99,
                 init_scale=0.1):
        super().__init__()
        self.num_centroids = num_centroids
        self.stop_gradient_codes = stop_gradient_codes
        self.ema_decay = ema_decay
        self.init_scale = init_scale
        self.codebook = None
        self.cluster_counts = None
        self.feature_means = None
        self.feature_stdev = None

    def get_num_centroids(self):
        return self.num_centroids

    def get_num_sections(self):
        return 1

    def create_codebook(self, flat_inputs):
        """Default codebook variable."""
        embedding_dim = flat_inputs.shape[-1]
        init_fn = torch.nn.init.kaiming_normal_  # Using Kaiming initialization as an approximation
        self.codebook = torch.nn.Parameter(
            torch.empty(self.num_centroids, embedding_dim))
        init_fn(self.codebook)
        return self.codebook

    def update_cluster_counts(self, encodings, train):
        """Track cluster utilization with an EMA counter."""
        counts = torch.sum(encodings,
                           dim=tuple(range(len(encodings.shape) - 1)))
        if self.cluster_counts is None:
            self.cluster_counts = torch.nn.Parameter(torch.ones(
                self.num_centroids),
                                                     requires_grad=False)
        if not train:
            return self.cluster_counts.data
        self.cluster_counts.data = self.ema_decay * self.cluster_counts.data + (
            1 - self.ema_decay) * counts
        return self.cluster_counts.data

    def update_mean_estimate(self, flat_inputs, train):
        """Update an EMA estimate of the feature means."""
        embedding_dim = flat_inputs.shape[-1]
        if self.feature_means is None:
            self.feature_means = torch.nn.Parameter(torch.zeros(embedding_dim),
                                                    requires_grad=False)
        new_observation = torch.mean(flat_inputs, dim=0)
        if train:
            self.feature_means.data = self.ema_decay * self.feature_means.data + (
                1 - self.ema_decay) * new_observation
        return self.feature_means.data

    def update_stdev_estimate(self, flat_inputs, train):
        """Update an EMA estimate of the feature standard deviation."""
        if self.feature_stdev is None:
            self.feature_stdev = torch.nn.Parameter(torch.std(flat_inputs),
                                                    requires_grad=False)
        new_observation = torch.std(flat_inputs)
        if train:
            self.feature_stdev.data = self.ema_decay * self.feature_stdev.data + (
                1 - self.ema_decay) * new_observation
        return self.feature_stdev.data


class VectorQuantizer(BaseQuantizer):
    """Vector Quantizer using L2-loss."""

    def __init__(self,
                 num_centroids,
                 commitment_loss=0.0,
                 demean=False,
                 rescale=False,
                 stop_gradient_codes=True,
                 ema_decay=0.99,
                 init_scale=0.1):
        super().__init__(num_centroids, stop_gradient_codes, ema_decay,
                         init_scale)
        self.commitment_loss = commitment_loss
        self.demean = demean
        self.rescale = rescale

    def loss(self, inputs, quantized):
        quant_loss = torch.square(quantized - inputs.detach())
        if self.commitment_loss > 0:
            encoder_loss = torch.square(quantized.detach() - inputs)
            quant_loss += self.commitment_loss * encoder_loss
        return quant_loss

    def forward(self, inputs, train):
        embedding_dim = inputs.shape[-1]
        flat_inputs = inputs.view(-1, embedding_dim)
        if self.demean:
            feature_means = self.update_mean_estimate(flat_inputs, train)
            flat_inputs -= feature_means
        stdev: Optional[torch.Tensor] = None
        feature_means: Optional[torch.Tensor] = None
        if self.rescale:
            stdev = self.update_stdev_estimate(flat_inputs, train)
            flat_inputs /= (stdev + 1e-8)
        codebook = self.create_codebook(flat_inputs)

        # Find nearest neighbor indices.
        distances = (
            torch.sum(torch.square(flat_inputs), dim=1, keepdim=True) -
            2 * torch.matmul(flat_inputs, codebook.T) +
            torch.sum(torch.square(codebook.T), dim=0, keepdim=True))
        nn_idx = torch.argmin(distances, dim=1)
        encodings = torch.nn.functional.one_hot(nn_idx,
                                                self.num_centroids).float()
        counts = self.update_cluster_counts(encodings, train)
        quantized = torch.matmul(encodings, codebook)
        quantization_loss = self.loss(flat_inputs, quantized)
        quantization_loss = quantization_loss.view(inputs.shape)

        if self.rescale:
            assert stdev is not None
            quantized *= (stdev + 1e-8)
        if self.demean:
            quantized += feature_means
        quantized = quantized.view(inputs.shape)

        nn_idx = nn_idx.view(inputs.shape[:-1])

        # Apply stop gradient to protect the encodings from downstream losses.
        quantized = inputs + (quantized - inputs).detach()

        # Expand the dimensions to match those of product quantizer, for interface
        # consistency. This can be seen as a product quantizer with just 1 section.
        nn_idx = nn_idx.unsqueeze(0)
        codebook_values = codebook.unsqueeze(0)

        if self.stop_gradient_codes:
            codebook_values = codebook_values.detach()

        return QuantizerOutputs(quantized, quantization_loss, nn_idx,
                                codebook_values, [counts])


class ResidualQuantizer(nn.Module):
    """A residual quantizer with explicitly passed sub-quantizers.

       Accepting a list allows using arbitrary quantizers (e.g., product quantizers)
       in sequence.
    """

    def __init__(self, quantizers, stop_gradient_codes=True):
        super().__init__()
        self.quantizers = torch.nn.ModuleList(quantizers)
        self.stop_gradient_codes = stop_gradient_codes

    def get_num_centroids(self):
        nc = [q.num_centroids for q in self.quantizers]
        assert (
            len(set(nc)) == 1
        ), 'Expected all quantizers to have the same number of centroids.'
        return nc[0]

    def get_num_sections(self):
        return len(self.quantizers)

    def forward(self, inputs, train=True):
        quantized: torch.Tensor = torch.tensor(0.0, device=inputs.device)
        quantization_loss: torch.Tensor = torch.tensor(0.0,
                                                       device=inputs.device)
        nn_idx, codebooks, counts = [], [], []
        embedding_dim = inputs.shape[-1]

        flat_inputs = inputs.view(-1, embedding_dim)
        residual = flat_inputs
        for quantizer in self.quantizers:
            quant_outputs = quantizer(residual, train)
            quantized += quant_outputs.quantized
            residual -= quant_outputs.quantized
            nn_idx.append(quant_outputs.nn_idx)
            codebooks.append(quant_outputs.codebook)
            quantization_loss += torch.mean(quant_outputs.quantization_loss)
            counts += quant_outputs.cluster_counts

        # Aggregate across 'sections' to get the following shapes:
        # quantized: [...].
        # nn_idx: [ns, ...].
        # codebook: [ns, nc, csz / ns].
        # Using non-homogenous quantizers means we can't concat the outputs.
        nn_idx = torch.cat(nn_idx, dim=0)
        nn_idx = nn_idx.view((len(self.quantizers), ) + inputs.shape[:-1])
        codebooks = torch.cat(codebooks, dim=0)
        if self.stop_gradient_codes:
            codebooks = codebooks.detach()
        quantized = quantized.view(inputs.shape)
        quantization_loss = quantization_loss.view(inputs.shape[:-1] + (1, ))
        return QuantizerOutputs(quantized, quantization_loss, nn_idx,
                                codebooks, counts)
