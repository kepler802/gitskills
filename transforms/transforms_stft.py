"""Transforms on the short time fourier transforms of wav samples."""

__author__ = 'Erdene-Ochir Tuguldur'

import random

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

from .transforms_wav import should_apply_transform

class ToSTFT(object):
    """Applies on an audio the short time fourier transform."""

    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        data['n_fft'] = self.n_fft
        data['hop_length'] = self.hop_length
        data['stft'] = librosa.stft(samples, n_fft=self.n_fft, hop_length=self.hop_length)
        data['stft_shape'] = data['stft'].shape
        return data

class StretchAudioOnSTFT(object):
    """Stretches an audio on the frequency domain."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        stft = data['stft']
        sample_rate = data['sample_rate']
        hop_length = data['hop_length']
        scale = random.uniform(-self.max_scale, self.max_scale)
        stft_stretch = librosa.core.phase_vocoder(stft, 1+scale, hop_length=hop_length)
        data['stft'] = stft_stretch
        return data

class TimeshiftAudioOnSTFT(object):
    """A simple timeshift on the frequency domain without multiplying with exp."""

    def __init__(self, max_shift=8):
        self.max_shift = max_shift

    def __call__(self, data):
        if not should_apply_transform():
            return data

        stft = data['stft']
        shift = random.randint(-self.max_shift, self.max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        stft = np.pad(stft, ((0, 0), (a, b)), "constant")
        if a == 0:
            stft = stft[:,b:]
        else:
            stft = stft[:,0:-a]
        data['stft'] = stft
        return data

class AddBackgroundNoiseOnSTFT(Dataset):
    """Adds a random background noise on the frequency domain."""

    def __init__(self, bg_dataset, max_percentage=0.45):
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage

    def __call__(self, data):
        if not should_apply_transform():
            return data

        noise = random.choice(self.bg_dataset)['stft']
        percentage = random.uniform(0, self.max_percentage)
        data['stft'] = data['stft'] * (1 - percentage) + noise * percentage
        return data

class FixSTFTDimension(object):
    """Either pads or truncates in the time axis on the frequency domain, applied after stretching, time shifting etc."""

    def __call__(self, data):
        stft = data['stft']
        t_len = stft.shape[1]
        orig_t_len = data['stft_shape'][1]
        if t_len > orig_t_len:
            stft = stft[:,0:orig_t_len]
        elif t_len < orig_t_len:
            stft = np.pad(stft, ((0, 0), (0, orig_t_len-t_len)), "constant")

        data['stft'] = stft
        return data

class ToMelSpectrogramFromSTFT(object):
    """Creates the mel spectrogram from the short time fourier transform of a file. The result is a 32x32 matrix."""

    def __init__(self, n_mels=32, deploy_param = True, fixed_pcen = False):
        self.n_mels = n_mels
        self.fixed_pcen =fixed_pcen
        self.deploy_param = deploy_param

    def pcen(self, x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False, last_state=None):
        frames = x.split(1, -2)
        m_frames = []

        for frame in frames:
            if last_state is None:
                last_state = frame
                m_frames.append(frame)
                continue
            if training:
                m_frame = ((1 - s) * last_state).add_(s * frame)
            else:
                m_frame = (1 - s) * last_state + s * frame
            last_state = m_frame
            m_frames.append(m_frame)
        M = torch.cat(m_frames, -2)
        if training:
            pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r
        else:
            pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
        return pcen_

    def __call__(self, data):
        stft = data['stft']
        sample_rate = data['sample_rate']
        n_fft = data['n_fft']
        if self.deploy_param:
            mel_basis = librosa.filters.mel(sample_rate, n_fft, self.n_mels, fmin=0, fmax = sample_rate // 2)
        else:
            mel_basis = librosa.filters.mel(sample_rate, n_fft, self.n_mels)
        s = np.dot(mel_basis, np.abs(stft)**2.0)
        if self.fixed_pcen:
            s = torch.tensor(s).unsqueeze(0).unsqueeze(0)
            s = s.transpose(2, 3)
            s = self.pcen(s)
            s = s.transpose(2, 3).squeeze().numpy()

        else:
            if self.deploy_param:
                s = librosa.power_to_db(s, amin=1e-6, ref=1.0, top_db = None)
            else:
                s = librosa.power_to_db(s, ref=np.max)

        data['mel_spectrogram'] = s
        return data

class DeleteSTFT(object):
    """Pytorch doesn't like complex numbers, use this transform to remove STFT after computing the mel spectrogram."""

    def __call__(self, data):
        del data['stft']
        return data

class AudioFromSTFT(object):
    """Inverse short time fourier transform."""

    def __call__(self, data):
        stft = data['stft']
        data['istft_samples'] = librosa.core.istft(stft, dtype=data['samples'].dtype)
        return data
