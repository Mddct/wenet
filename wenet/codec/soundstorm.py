import math
import torch

from wenet.transformer.encoder import ConformerEncoder
from wenet.utils.class_utils import WENET_NORM_CLASSES
from wenet.utils.mask import make_non_pad_mask, mask_finished_preds


def schedule(ratio, method="cosine"):
    """
    Generates a mask rate by scheduling mask functions R.

    Args:
        ratio: The uniformly sampled ratio [0, 1) as input.
        total_unknown: The total number of tokens that can be masked out. For
          example, in MaskGIT, total_unknown = 256 for 256x256 images and 1024 for
          512x512 images.
        method: implemented functions are ["uniform", "cosine"]

    Returns:
        The mask rate (torch.Tensor float).
    """
    if method == "uniform":
        mask_ratio = 1. - ratio
    elif "pow" in method:
        exponent = float(method.replace("pow", ""))
        mask_ratio = 1. - ratio**exponent
    elif method == "cosine":
        mask_ratio = torch.cos(math.pi / 2. * ratio)
    # Clamps mask into [epsilon, 1)
    mask_ratio = torch.clamp(mask_ratio, 1e-6, 1.)
    return mask_ratio


# https://github.com/lucidrains/soundstorm-pytorch/blob/main/soundstorm_pytorch/soundstorm.py#L86
def get_mask_subset_prob(mask: torch.Tensor,
                         prob: torch.Tensor,
                         min_mask: int = 0):
    batch, seq, device = *mask.shape, mask.device

    num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
    logits = torch.rand((batch, seq), device=device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim=-1).argsort(dim=-1).float()

    num_padding = (~mask).sum(dim=-1, keepdim=True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask


class SoundStorm(torch.nn.Module):
    """ https://arxiv.org/pdf/2305.09636
    """

    def __init__(
        self,
        hidden,
        conformer: ConformerEncoder,
        num_semantic_tokens: int,
        codebook_size: int,
        num_quantizers: int = 8,
        grouped_quantizers: int = 1,
    ) -> None:
        super().__init__()
        self.conformer = conformer
        # 1 is mask id for current q
        total_codes = (codebook_size + 1) * num_quantizers * grouped_quantizers
        self.mask_id = codebook_size + 1
        self.semantic_embeding = torch.nn.Embedding(num_semantic_tokens,
                                                    hidden)
        self.codec_embeding = torch.nn.Embedding(
            total_codes,
            hidden,
        )
        self.embedding_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden * grouped_quantizers, hidden),
            torch.nn.LayerNorm(
                hidden)) if grouped_quantizers > 1 else torch.nn.Identity()

        total_quantizers = num_quantizers * grouped_quantizers
        output_size = conformer.output_size()
        self.heads = torch.nn.Linear(output_size,
                                     output_size * total_quantizers)

        self.total_quantizers = total_quantizers
        self.num_quantizers = num_quantizers
        self.grouped_quantizers = grouped_quantizers

        # TODO: xavier init
        self.to_logits_weight = torch.nn.Parameter(
            torch.randn(total_quantizers, output_size, codebook_size))
        self.to_logits_bias = torch.nn.Parameter(
            torch.randn(total_quantizers, codebook_size))

        self.register_buffer(
            'quantizer_offsets',
            torch.arange(total_quantizers) * (codebook_size + 1),
            persistent=False,
        )

    def _mask_acoustics(self, acoustics: torch.Tensor, mask: torch.Tensor):
        device = acoustics.device
        B, T, _ = acoustics.size()
        seq_mask = mask
        min_seq_len = seq_mask.sum(-1).amin()
        t = torch.randint(0, min_seq_len, (), device=acoustics.device)
        t_mask = mask[:, t:]

        rand_times = torch.empty(B, device=acoustics.device).uniform_(0, 1)
        p = schedule(rand_times)

        t_mask = get_mask_subset_prob(t_mask, p)

        q = torch.randint(
            0, self.total_quantizers,
            (), device=acoustics.device) * self.grouped_quantizers

        masked = torch.where(t_mask, self.mask_id, acoustics[:, t:, q])
        masked = torch.cat((acoustics[:, :t, q], masked), dim=1).unsqueeze(2)

        masked = torch.cat(
            (acoustics[:, :, :q], masked, acoustics[:, :, q + 1:]), dim=2)
        masked[:, :, q + 1:] = self.mask_id

        prompt_mask = torch.full((B, t), False, device=device)
        lower_quantizers_mask = torch.full((B, T, q), False, device=device)
        upper_quantizers_mask = torch.full((B, T, self.num_quantizers - q - 1),
                                           True,
                                           device=device)
        upper_quantizers_mask[:, :t, :] = False
        mask = torch.cat((prompt_mask, t_mask),
                         dim=1).unsqueeze(2)  # (B, T,12)
        mask = torch.cat((lower_quantizers_mask, mask, upper_quantizers_mask),
                         dim=2)
        mask[:, :, q + 1:] = False
        mask = mask * seq_mask.squeeze(1)
        mask = mask.view(B, -1)
        return masked, mask

    def forward(self, batch: dict, device: torch.device):
        # NOTE(Mddct) we assume semantic.size() == acoustic[:,:,0].size() for now
        semantics = batch['semantics'].to(device)  # [B,T]
        acoustics = batch['acoustics'].to(device)  # [B,T,Q]
        lengths = batch['lengths'].to(device)

        targets = acoustics
        acoustics = acoustics + self.quantizer_offsets

        # TODO: assert
        # TODO: speech prompt: spk embedding, 3s prompts, 0-15s prompts
        # TODO: loss
        # TODO: streaming
        # TODO: acc

        B, T, _ = acoustics.size()  # (B, T, Q)
        mask = make_non_pad_mask(lengths)  # (B, T)

        semantics_emb = self.semantic_embeding(semantics)  # (B,T,E)

        masked_acoustics, mask = self._mask_acoustics(acoustics, mask)

        masked_acoustics = masked_acoustics.view(
            B, T, -1, self.num_quantizers)  # (B,T,G,q)
        acoustic_emb = self.codec_embeding(masked_acoustics)  # (B,T,G,q,E)
        acoustic_emb = self.embedding_proj(
            acoustic_emb.sum(-2).view(B, T, -1))  # (B,T,E)

        x = acoustic_emb + semantics_emb
        x, _ = self.conformer(x, lengths)

        heads = self.heads(x)  # (B, T, D*total_quantizers)
        heads = heads.view(B, T, self.total_quantizers, -1)  # (B,T,-1, D)

        logits = torch.einsum('bngd,gdl->bngl', heads, self.to_logits_weight)
        logits += self.to_logits_bias
        c = logits.size(-1)
        logits = logits.contiguous().view(B, -1, c)
        loss = torch.nn.functional.cross_entropy(logits[mask],
                                                 targets.view(B, -1)[mask])
        return {"loss": loss}

    @torch.no_grad()
    def generate(self,
                 semantics: torch.Tensor,
                 semantic_lens: torch.Tensor,
                 steps=8):
        # TODO: speech prompt: spk embedding, 3s prompts, 0-15s prompts
        batch_size, seq_length = semantics.size()
        device = semantics.device
        mask = torch.ones((batch_size, seq_length, self.num_quantizers),
                          device=device)
        masked = mask * self.mask_id
        mask = mask.bool()

        prompt_mask = mask.clone()
        seq_mask = make_non_pad_mask(semantics)
        seq_mask_with_quantizer = seq_mask.unsqueeze(2).repeat(
            1, 1, self.total_quantizers)

        semtntic_embeddings = self.semantic_embeding(semantics)
        times = torch.linspace(0., 1., steps + 1, device=device)
        rand_mask_probs = schedule(times).unsqueeze(1)  # (B,1)
        all_mask_num_tokens = (rand_mask_probs * semantic_lens).long()

        for q in range(self.total_quantizers):
            for i, mask_num_tokens in enumerate(all_mask_num_tokens):
                masked_input = rearrange(masked, 'b n q -> b (n q)')
                logits = self.net(masked_input.long(),
                                  mask=seq_mask,
                                  cond=cond_tokens)
                logits = top_k(logits, thres=topk_pres)
                temperature = 1.0 * i / steps
                sampled_ids = gumbel_sample(logits, temperature=temperature)
                if q >= num_full_sampling_levels:
                    masked[:, :, q:] = rearrange(sampled_ids,
                                                 'b (n q) -> b n q',
                                                 q=self.num_quantizers)[:, :,
                                                                        q:]
                    mask[:, :, q] = False
                    continue

                masked = torch.where(
                    mask,
                    rearrange(sampled_ids,
                              'b (n q) -> b n q',
                              q=self.num_quantizers), masked)
                if (mask_num_tokens == 0).all():
                    continue

                scores = 1 - logits.softmax(dim=-1)
                scores = scores.gather(2, rearrange(sampled_ids,
                                                    'b n -> b n 1'))
                scores = rearrange(scores, 'b n 1 -> b n')
                mask = torch.zeros_like(scores,
                                        dtype=torch.bool,
                                        device=device)
                mask_value = -torch.finfo(scores.dtype).max
                scores = scores.masked_fill(~seq_mask_with_quantizer,
                                            mask_value)
                scores_sorted = scores.argsort(dim=-1, descending=True)
                mask_num_tokens = rearrange(mask_num_tokens, 'b -> b 1')
                mask_tokens = scores_sorted[:, :mask_num_tokens]
                rows = torch.arange(
                    mask_tokens.size(0)).unsqueeze(-1).expand_as(mask_tokens)
                mask[rows, mask_tokens] = True
                mask = rearrange(mask,
                                 'b (n q) -> b n q',
                                 q=self.num_quantizers)
                mask[:, :, (q + 1):] = True
                mask = mask & prompt_mask

                masked = masked.masked_fill(mask, self.mask_id)
        output = torch.cat(
            [semantic_tokens.unsqueeze(-1), masked[:, -seq_length:]], axis=-1)
        return output.detach().long()