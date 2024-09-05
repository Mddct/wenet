from dataclasses import dataclass
import os
from typing import List, Tuple

import torch
from torch.nn import DataParallel, parallel
import torchaudio
from wenet.codec.vocos_low_latency.dataset import init_train_dataset, multihost_dataloader
from wenet.codec.vocos_low_latency.discriminators import (
    MultiPeriodDiscriminator, MultiResolutionDiscriminator)
from wenet.codec.vocos_low_latency.vocos_my import (Vocosv1, vocos_config)
from wenet.utils.checkpoint import save_state_dict_and_infos
from wenet.utils.common import tensor_to_scalar
from wenet.utils.scheduler import WarmupLR
from wenet.utils.train_utils import init_distributed, init_summarywriter


def compute_discriminator_loss(disc_real_outputs: List[torch.Tensor],
                               disc_generated_outputs: List[torch.Tensor]):
    loss = torch.zeros(1,
                       device=disc_real_outputs[0].device,
                       dtype=disc_real_outputs[0].dtype)
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean(torch.clamp(1 - dr, min=0))
        g_loss = torch.mean(torch.clamp(1 + dg, min=0))
        loss += r_loss + g_loss
        r_losses.append(r_loss)
        g_losses.append(g_loss)

    return loss, r_losses, g_losses


def compute_generator_loss(
    disc_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    loss = torch.zeros(1,
                       device=disc_outputs[0].device,
                       dtype=disc_outputs[0].dtype)
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean(torch.clamp(1 - dg, min=0))
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def compute_feature_matching_loss(
        fmap_r: List[List[torch.Tensor]],
        fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss


def init_train_iter(
    data_list_file,
    epoch,
    num_workers,
    prefetch,
    pin_memory,
    seed=2024,
):
    dataset = init_train_dataset(data_list_file, epoch)
    dataloader = multihost_dataloader(dataset, num_workers, prefetch,
                                      pin_memory, seed)
    return dataloader


class MelSpecReconstructionLoss(torch.nn.Module):
    """
    L1 distance between the mel-scaled magnitude spectrograms of the ground truth sample and the generated sample
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
    ):
        super().__init__()

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )

    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        mel_hat = torch.log(torch.clip(self.mel_spec(y_hat), min=1e-7))
        mel = torch.log(torch.clip(self.mel_spec(y), min=1e-7))
        loss = torch.nn.functional.l1_loss(mel, mel_hat)

        return loss


@dataclass
class TrainConfig:
    config: vocos_config

    # pretrain_mel_steps:
    pretrain_mel_steps = 0
    mrd_loss_coeff = 0.1
    mel_loss_coeff = 45

    # optimizer config
    opt_disc_config = {"lr": 0.001, 'betas': (0.8, 0.9)}
    opt_gen_config = {"lr": 0.001, 'betas': (0.8, 0.9)}

    # scheduler conf
    disc_scheduler_config = {'warmup_steps': 25000}
    gen_scheduler_config = {'warmup_steps': 25000}


@dataclass
class TrainState:
    model: Vocosv1
    multiperioddisc: MultiPeriodDiscriminator
    multiresddisc: MultiResolutionDiscriminator

    scheduler_d: torch.optim.lr_scheduler._LRScheduler
    scheduler_g: torch.optim.lr_scheduler._LRScheduler
    optimizer_d: torch.optim.Optimizer
    optimizer_g: torch.optim.Optimizer

    # global_step
    global_step = 0

    def __call__(self, input, input_lens):
        return self.model(input, input_lens)


def create_state(model, multiperioddisc, multiresddisc, opt_disc, opt_gen,
                 opt_d_scheduler, opt_g_scheduler):
    return TrainState(model=model,
                      multiperioddisc=multiperioddisc,
                      multiresddisc=multiresddisc,
                      optimizer_d=opt_disc,
                      optimizer_g=opt_gen,
                      scheduler_d=opt_d_scheduler,
                      scheduler_g=opt_g_scheduler)


def train_step(batch,
               state: TrainState,
               train_config: TrainConfig,
               mel_loss_fn,
               device,
               global_step: int = 0,
               **kwargs):

    mels, mels_lens = batch['mels'].to(device), batch['mels_lens'].to(device)
    audio = batch['wavs'].to(device)
    metrics = {}
    for idx in [0, 1]:
        # 1 train discriminator
        if idx == 0 and global_step >= train_config.pretrain_mel_steps:
            with torch.no_grad():
                audio_hat, audio_hat_lens = state(mels, mels_lens)

            real_score_mp, gen_score_mp, _, _ = state.multiperioddisc(
                y=audio,
                y_hat=audio_hat,
                **kwargs,
            )
            real_score_mrd, gen_score_mrd, _, _ = state.multiresddisc(
                y=audio,
                y_hat=audio_hat,
                **kwargs,
            )
            loss_mp, loss_mp_real, _ = compute_discriminator_loss(
                disc_real_outputs=real_score_mp,
                disc_generated_outputs=gen_score_mp,
            )
            loss_mrd, loss_mrd_real, _ = compute_discriminator_loss(
                disc_real_outputs=real_score_mrd,
                disc_generated_outputs=gen_score_mrd,
            )
            loss_mp /= len(loss_mp_real)
            loss_mrd /= len(loss_mrd_real)
            loss = loss_mp + train_config.mrd_loss_coeff * loss_mrd

            metrics["discriminator/total"] = loss
            metrics["discriminator/multi_period_loss"] = loss_mp
            metrics["discriminator/multi_res_loss"] = loss_mrd
            state.optimizer_d.zero_grad()
            loss.backward()
            state.scheduler_d.step()

        else:
            audio_hat, audio_hat_lens = state(mels, mels_lens)
            if global_step >= train_config.pretrain_mel_steps:
                _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = state.multiperioddisc(
                    y=audio,
                    y_hat=audio_hat,
                    **kwargs,
                )
                _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = state.multiresddisc(
                    y=audio,
                    y_hat=audio_hat,
                    **kwargs,
                )
                loss_gen_mp, list_loss_gen_mp = compute_generator_loss(
                    disc_outputs=gen_score_mp)
                loss_gen_mrd, list_loss_gen_mrd = compute_generator_loss(
                    disc_outputs=gen_score_mrd)
                loss_gen_mp = loss_gen_mp / len(list_loss_gen_mp)
                loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)
                loss_fm_mp = compute_feature_matching_loss(
                    fmap_r=fmap_rs_mp, fmap_g=fmap_gs_mp) / len(fmap_rs_mp)
                loss_fm_mrd = compute_feature_matching_loss(
                    fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)

            else:
                loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = 0

            mel_loss = mel_loss_fn(audio_hat, audio)
            loss = (loss_gen_mp + train_config.mrd_loss_coeff * loss_gen_mrd +
                    loss_fm_mp + train_config.mrd_loss_coeff * loss_fm_mrd +
                    train_config.mel_loss_coeff * mel_loss)
            state.optimizer_g.zero_grad()
            loss.backward()
            state.scheduler_g.step()

            metrics["generator/total_loss"] = loss
            metrics["generator/mel_loss"] = mel_loss
            metrics["generator/multi_period_loss"] = loss_gen_mp
            metrics["generator/multi_res_loss"] = loss_gen_mrd
            metrics["generator/feature_matching_mp"] = loss_fm_mp
            metrics["generator/feature_matching_mrd"] = loss_fm_mrd

    return metrics


def save_checkpoint(ckpt_dir, train_state: TrainState):
    # "chpt_dir/step_1200/model.pt"
    # 'ckpt_dir/step_1200/opt.pt'

    global_step = train_state.global_step

    model_opt_dir = os.path.join(ckpt_dir, 'step_' + str(global_step))
    os.makedirs(model_opt_dir)

    def _save_checkpoint(model_or_opt, path: str, name: str):
        if isinstance(model_or_opt, torch.nn.DataParallel):
            state_dict = model_or_opt.module.state_dict()
        elif isinstance(model_or_opt,
                        torch.nn.parallel.DistributedDataParallel):
            state_dict = model_or_opt.module.state_dict()
        else:
            state_dict = model_or_opt.state_dict()
        torch.save({name: state_dict}, path)

    # model: generator
    model_path = os.path.join(model_opt_dir, 'model.pt')
    _save_checkpoint(train_state.model, model_path, 'model')

    # discriminattor
    period_d_path = os.path.join(model_opt_dir, 'model_period_d.pt')
    _save_checkpoint(train_state.multiperioddisc, period_d_path, 'period_d')
    res_d_path = os.path.join(model_opt_dir, 'model_res_d.pt')
    _save_checkpoint(train_state.multiresddisc, res_d_path, 'res_g')

    # optimizer
    opt_g_path = os.path.join(model_opt_dir, 'opt_g.pt')
    _save_checkpoint(train_state.optimizer_g, opt_g_path, 'opt_g')

    opt_d_path = os.path.join(model_opt_dir, 'opt_d.pt')
    _save_checkpoint(train_state.optimizer_d, opt_d_path, 'opt_d')

    # scheduler
    opt_d_path = os.path.join(model_opt_dir, 'scheduler_d.pt')
    _save_checkpoint(train_state.scheduler_d, opt_d_path, 'scheduler_d')

    opt_d_path = os.path.join(model_opt_dir, 'scheduler_g.pt')
    _save_checkpoint(train_state.scheduler_g, opt_g_path, 'scheduler_g')


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--data_list', required=True, help='train data file')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        choices=["cpu", "npu", "cuda"])
    parser.add_argument('--model_dir', required=True, type=str)
    parser.add_argument('--tensorboard_dir', required=True, type=str)
    args = parser.parse_args()
    return args


def main():
    # TODO: args
    args = get_args()
    args.train_engine = 'torch_ddp'
    args.dist_backend = 'nccl'
    _, _, rank = init_distributed(args)

    # init dataset
    train_iter = init_train_iter(args.data_list,
                                 100,
                                 args.num_workers,
                                 args.prefetch,
                                 args.pin_memory,
                                 seed=2024)
    device = torch.device(args.device)
    # init modelg
    model_config = vocos_config()
    model_config.input_size = 128
    model_config.n_fft = 2048
    model_config.hop_length = 640
    model = Vocosv1(model_config)
    multiperioddisc = MultiPeriodDiscriminator()
    multiresddisc = MultiResolutionDiscriminator()

    # train config
    config = TrainConfig(config=model_config)
    # Tensorboard
    writer = init_summarywriter(args)

    disc_params = [
        {
            "params": multiperioddisc.parameters()
        },
        {
            "params": multiresddisc.parameters()
        },
    ]
    gen_params = [
        {
            "params": model.parameters()
        },
    ]
    opt_disc = torch.optim.AdamW(disc_params, **config.opt_disc_config)
    opt_gen = torch.optim.AdamW(gen_params, **config.opt_gen_config)
    scheduler_disc = WarmupLR(opt_disc, **config.disc_scheduler_config)
    scheduler_gen = WarmupLR(opt_gen, **config.gen_scheduler_config)

    model.to(device)
    multiperioddisc.to(device)
    multiresddisc.to(device)

    import torch.distributed as dist
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=False)
        multiperioddisc = torch.nn.parallel.DistributedDataParallel(
            multiperioddisc, find_unused_parameters=False)
        multiresddisc = torch.nn.parallel.DistributedDataParallel(
            multiresddisc, find_unused_parameters=False)
    train_state = create_state(
        model,
        multiperioddisc,
        multiresddisc,
        opt_disc,
        opt_gen,
        scheduler_disc,
        scheduler_gen,
    )

    # TODO: restore_ckpt(args)

    global_step = 0
    mel_loss_fn = MelSpecReconstructionLoss().to(device)
    for i, batch in enumerate(train_iter):
        metric = train_step(batch, train_state, config, mel_loss_fn, device,
                            global_step)

        # write to tensorboard
        if rank == 0:
            for name, value in metric.items():
                writer.add_scalar(name, tensor_to_scalar(value), i)

        # log interval
        if i % 100 == 0:
            log_str = f'iter i'
            for name, value in metric:
                log_str += '| {:.6f} '.format(tensor_to_scalar(value))
            print(log_str + "| rank {}".format(rank))

        # save interval
        if i % 2000 == 0:
            save_checkpoint(args.model_path, train_state)


if __name__ == '__main__':
    main()
