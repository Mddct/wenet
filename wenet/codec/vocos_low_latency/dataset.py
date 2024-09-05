from functools import partial

import torch
from torch.utils.data import DataLoader
from wenet.dataset.datapipes import WenetRawDatasetSource
from wenet.dataset.processor import decode_wav, parse_json, resample
import torchaudio

import numpy as np


def mix_to_mono(sample):
    wav = sample['wav']
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    sample['wav'] = wav
    return sample


def gain_for_train(sample):

    wav = sample['wav']
    sr = sample['sample_rate']
    gain = np.random.uniform(-1, -6)
    wav, _ = torchaudio.sox_effects.apply_effects_tensor(
        wav, sr, [["norm", f"{gain:.2f}"]])
    sample['wav'] = wav
    return sample


def trim_to_n_samples(sample, n_samples=32000 * 3):
    wav = sample['wav']
    if wav.size(-1) < n_samples:
        pad_length = n_samples - wav.size(-1)
        padding_tensor = wav.repeat(1, 1 + pad_length // wav.size(-1))
        new_wav = torch.cat((wav, padding_tensor[:, :pad_length]), dim=1)
    else:
        start = np.random.randint(low=0, high=wav.size(-1) - n_samples + 1)
        new_wav = wav[:, start:start + n_samples]

    sample['wav'] = new_wav
    sample['wav_lens'] = new_wav.size(-1)
    return sample


def compute_features(sample, feature_extractor):
    wav = sample['wav']
    mel = feature_extractor(wav)
    sample['mel'] = mel.transpose(1, 2)  # [1,T,D]
    sample['mel_lens'] = sample['mel'].size(1)
    return sample


class MelSpectrogramFeatures:

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

    def __call__(self, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        mel = torch.log(torch.clip(self.mel_spec(y), min=1e-7))
        return mel


def batching(samples):
    keys = [sample['key'] for sample in samples]
    wavs = [sample['wav'] for sample in samples]
    wavs_lens = [sample['wav_lens'] for sample in samples]
    mels_lens = [sample['mel_lens'] for sample in samples]
    mels = [sample['mel'] for sample in samples]

    wavs_tensor = torch.cat(wavs, dim=0)
    mels_tensor = torch.cat(mels, dim=0)
    wavs_lens_tensor = torch.tensor(wavs_lens, dtype=torch.int64)
    mels_lens_tensor = torch.tensor(mels_lens, dtype=torch.int64)
    return {
        "mels": mels_tensor,
        "wavs": wavs_tensor,
        "keys": keys,
        "wavs_lens": wavs_lens_tensor,
        "mels_lens": mels_lens_tensor,
    }


def init_train_dataset(data_list_file, max_iters: int):
    dataset = WenetRawDatasetSource(data_list_file,
                                    shuffle=True,
                                    shuffle_size=1000000,
                                    cycle=max_iters,
                                    partition=True)

    dataset = dataset.map(parse_json)
    dataset = dataset.map(decode_wav)
    dataset = dataset.map(mix_to_mono)
    dataset = dataset.map(gain_for_train)
    dataset = dataset.map(partial(resample, resample_rate=32000))
    dataset = dataset.map(trim_to_n_samples)

    feature_extractor = MelSpectrogramFeatures(sample_rate=32000,
                                               n_fft=2048,
                                               hop_length=640,
                                               n_mels=128)

    dataset = dataset.map(
        partial(compute_features, feature_extractor=feature_extractor))

    dataset = dataset.batch(100, wrapper_class=batching)

    return dataset


def multihost_dataloader(
    dataset,
    num_workers,
    prefetch,
    pin_memory=False,
    seed=2024,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(dataset,
                            batch_size=None,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            prefetch_factor=prefetch,
                            persistent_workers=True,
                            generator=generator)

    return dataloader
