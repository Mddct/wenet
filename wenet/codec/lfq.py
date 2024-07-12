import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def compute_entropy_loss(
    affinity,
    mask=None,
    loss_type="softmax",
    temperature=1.0,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
):
    """Calculate the entropy loss

    Args:
        affinity: shape (B, T, D)
        mask: shape (B, T)
    """
    flat_affinity = affinity.view(-1, affinity.shape[-1])
    if mask is not None:
        mask = mask.view(-1).unsqueeze(1)
        flat_affinity = flat_affinity.masked_fill(mask == 0, float('-inf'))
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)

    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = torch.argmax(flat_affinity, dim=-1)
        onehots = F.one_hot(codes, num_classes=flat_affinity.shape[-1]).type(
            flat_affinity.dtype)
        onehots = probs - (probs - onehots).detach()  # stop gradient
        target_probs = onehots
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))

    if mask is not None:
        target_probs = target_probs * mask
        log_probs = log_probs * mask
        avg_probs = torch.sum(target_probs, dim=0) / mask.sum()
        sample_entropy = target_probs * log_probs
        sample_entropy = -torch.sum(sample_entropy * mask) / mask.sum()
    else:
        avg_probs = torch.mean(target_probs, dim=0)
        sample_entropy = -torch.mean(
            torch.sum(target_probs * log_probs, dim=-1))
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))

    loss = (sample_minimization_weight *
            sample_entropy) - (batch_maximization_weight * avg_entropy)
    return loss


def bits_codebook(size, dim):
    all_codes = torch.arange(size)
    mask = 2**torch.arange(dim)
    bits = (all_codes.unsqueeze(1) & mask).ne(0)
    return bits.float()


class LFQQuantizer(torch.nn.Module):

    def __init__(self,
                 hidden,
                 codebook_size: int,
                 commitment_weight: float = 0.25,
                 entropy_loss_weight: float = 0.1,
                 latent_normalized: bool = False,
                 sample_minimization_weight: float = 1.0,
                 batch_maximization_weight: float = 1.0) -> None:
        super().__init__()
        assert math.log2(codebook_size).is_integer()
        self.commitment_weight = commitment_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.latent_normalized = latent_normalized

        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

        self.num_codebooks = 1
        self.codebook_size = codebook_size
        self.codebook_dim = int(math.log2(codebook_size))
        self.project = torch.nn.Linear(hidden,
                                       self.codebook_dim * self.num_codebooks)

        # 0 or 1
        bits = bits_codebook(self.codebook_size, self.codebook_dim)
        # turn to {-1, 1}
        self.register_buffer('codebook', 2 * bits - 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.project(x)

        if self.latent_normalized:
            x = torch.nn.functional.normalize(x, dim=-1)

        B, T, _ = x.size()
        x = x.view(B, T, self.num_codebooks, -1)

        tokens, quantized = self.quantize(x)
        e_latent_loss = (quantized.detach() - x)**2
        if mask is not None:
            e_latent_loss = e_latent_loss * mask.unsqueeze(-1).unsqueeze(-1)
            e_latent_loss = e_latent_loss.sum() / mask.sum()
        else:
            e_latent_loss = torch.mean(e_latent_loss)
        e_latent_loss = e_latent_loss * self.commitment_weight

        logits = 2 * torch.einsum("... i d, j d -> ... i j", x, self.codebook)
        entropy_loss = compute_entropy_loss(
            -logits,
            mask=mask,
            sample_minimization_weight=self.sample_minimization_weight,
            batch_maximization_weight=self.batch_maximization_weight)
        entropy_loss *= self.entropy_loss_weight

        loss = e_latent_loss + entropy_loss
        if self.training:
            quantized = x + (quantized - x).detach()
        return {
            "loss": loss,
            "entropy_loss": entropy_loss,
            "commitment_loss": e_latent_loss,
            "tokens": tokens,
            "quantized": quantized,
        }

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized = torch.where(z > 0, 1, -1)
        indices = (torch.arange(z.size(-1), dtype=torch.float32)).to(z.device)
        tokens = torch.sum(2**indices * (z > 0).float(), dim=-1)
        return tokens, quantized
