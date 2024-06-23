"""Transforms on raw wav samples."""

__author__ = 'Yuan Xu'

import random
import numpy as np
import librosa

import torch
from torch.utils.data import Dataset
import torchaudio
import os
from pydub import AudioSegment
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

def should_apply_transform(prob=0.5):
    """Transforms are only randomly applied with the given probability."""
    return random.random() < prob

class LoadAudio(object):
    """Loads an audio into a numpy array."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, data):
        path = data['path']
        if path:
            samples, sample_rate = librosa.load(path, self.sample_rate)
            if sample_rate != 8000:
                print("!!!!!!!!!!!!!!")
                samples = librosa.resample(samples, sample_rate, 8000) 
                samples *= 0.4
                sample_rate = 8000

        else:
            # silence
            sample_rate = self.sample_rate
            samples = np.zeros(sample_rate, dtype=np.float32)
        data['samples'] = samples
        data['sample_rate'] = sample_rate
        return data

class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1):
        self.time = time

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(self.time * sample_rate)
        if length < len(samples):
            if data['target'] == 0:
                st = random.randint(0, len(samples)-length-1)
            else:
                st = (len(samples)-length)//2
            data['samples'] = samples[st:st+length]
        elif length > len(samples):
            left = random.randint(0, length-len(samples)-1)
            data['samples'] = np.pad(samples, (left, length - len(samples)-left), "constant")
        return data

class SplitAudio(object):

    def __init__(self, time=1):
        self.time = time

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(self.time * sample_rate)

        if length < len(samples):

            if "common_voice" in data['path']:
                split_info = data['path'].split("/")[-1][:-4].split("#")[-1].split("_")
                split_info = [float(i) for i in split_info]

                if split_info[-1] - split_info[0] > self.time:
                    speed_fac = (split_info[-1] - split_info[0])/self.time

                    samples = samples[int(split_info[0]*sample_rate) : int(split_info[-1]*sample_rate)]
                    samples = np.interp(np.arange(0, len(samples), speed_fac), np.arange(0,len(samples)), samples).astype(np.float32)
                    data['samples'] = samples[:length]

                else:

                    left = max(0, split_info[-1]-2)
                    right = min(len(samples)/sample_rate - 2, split_info[0])

                    st = int(random.uniform(left, right) * sample_rate)  
                    data['samples'] =  samples[st: st + length]                

        return data

class FixAudioLengthTail(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1):
        self.time = time

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(self.time * sample_rate)
        if length < len(samples):
            if data['target'] == 0:
                st = random.randint(0, len(samples)-length-1)
            else:
                st = (len(samples)-length)//2
            data['samples'] = samples[-length:]
        elif length > len(samples):
            left = random.randint(0, length-len(samples)-1)
            data['samples'] = np.pad(samples, (left, length - len(samples)-left), "constant")
        return data

class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""

    def __init__(self, amplitude_range=(0.7, 1.1)):
        self.amplitude_range = amplitude_range

    def __call__(self, data):
        if not should_apply_transform():
            return data

        if "train_yourTTS" in data['path']:
            print("train_yourTTS!!!!!!!!!!!!!!!!!!!!!!")

            for i in range(10, 0, -1):
                if np.sum(data['samples'][int(-i*0.1*8000):]) == 0:
                    data['samples'] = data['samples'][:int(-i*0.1*8000)]
                    break

            data['samples'] = data['samples'] * random.uniform(0.3, 0.4)
            return data
        
        data['samples'] = data['samples'] * random.uniform(*self.amplitude_range)
        return data

class ChangeSpeedAndPitchAudio(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""

    def __init__(self, min_scale = 1.0, max_scale=1.3):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data['samples']
        sample_rate = data['sample_rate']
        # scale = random.uniform(-self.max_scale, self.max_scale)
        # speed_fac = 1.0  / (1 + scale)
        speed_fac = random.uniform(self.min_scale, self.max_scale)
        data['samples'] = np.interp(np.arange(0, len(samples), speed_fac), np.arange(0,len(samples)), samples).astype(np.float32)

        # torchaudio.save(os.path.join("tmp_test/train_aug", data['path'].split("/")[-1]),torch.from_numpy(data['samples'][None, :]),  sample_rate)
        return data


class ChangeSpeedPydub(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""

    def __init__(self, max_scale=1.5):
        self.max_scale =max_scale
        assert max_scale > 1.0

    def __call__(self, data):
        if not should_apply_transform():
            return data

        sample_width = 2
        samples = data['samples']
        sample_rate = data['sample_rate']

        speed_fac = random.uniform(1.0, self.max_scale)

        audio = np.array(samples*(2**(sample_width * 8 - 1)), dtype=np.int16)

        try:
            audio = AudioSegment(audio.tobytes(), sample_width=sample_width, frame_rate = sample_rate, channels = 1)


            audio = audio.speedup(playback_speed = speed_fac)
            audio = np.array(audio.get_array_of_samples()) / 2**(sample_width * 8 - 1)
            audio = audio.astype(np.float32)

            data['samples'] = audio

        except:
            print("samples shape: ", samples.shape)
            print(data['path'])


        # torchaudio.save(os.path.join("tmp_test/train_aug", data['path'].split("/")[-1]),torch.from_numpy(data['samples'][None, :]),  sample_rate)
        return data

class StretchAudio(object):
    """Stretches an audio randomly."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        scale = random.uniform(-self.max_scale, self.max_scale)
        data['samples'] = librosa.effects.time_stretch(data['samples'], 1+scale)
        return data

class TimeshiftAudio(object):
    """Shifts an audio randomly."""

    def __init__(self, max_shift_seconds=0.2):
        self.max_shift_seconds = max_shift_seconds

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data['samples']
        sample_rate = data['sample_rate']
        max_shift = (sample_rate * self.max_shift_seconds)
        shift = random.randint(-max_shift, max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        samples = np.pad(samples, (a, b), "constant")
        data['samples'] = samples[:len(samples) - a] if a else samples[b:]
        return data

class AddBackgroundNoise(Dataset):
    """Adds a random background noise."""

    def __init__(self, bg_dataset, max_percentage=0.45):
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data['samples']
        noise = random.choice(self.bg_dataset)['samples']
        percentage = random.uniform(0, self.max_percentage)
        data['samples'] = samples * (1 - percentage) + noise * percentage
        return data

class ToMelSpectrogram(object):
    """Creates the mel spectrogram from an audio. The result is a 32x32 matrix."""

    def __init__(self, n_mels=32, deploy_param = True, fixed_pcen = False):
        self.n_mels = n_mels
        self.fixed_pcen = fixed_pcen
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
        samples = data['samples']
        sample_rate = data['sample_rate']
        if self.deploy_param:
            s = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=self.n_mels, n_fft=1024,hop_length=128, fmin=0, fmax = sample_rate // 2)
        else:
            s = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=self.n_mels, n_fft=1024,hop_length=128)

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

class ExtractFeature(object):
    def __init__(self, sample_rate, num_mel=40, window='hann', center=True, pad_mode='reflect',
                ref= 1.0, n_fft=1024, amin=1e-6, top_db = None, hop_length=128, fmin=0):
        super(ExtractFeature, self).__init__()
        n_mels = num_mel
        win_length = n_fft
        fmax = sample_rate // 2
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True,is_log=True)

    def __call__(self, data):

        x = torch.from_numpy(data['samples']).unsqueeze(0)
        x = self.spectrogram_extractor(x)
        # print('outspec:',x[0,0,0,:10])
        logmel_spec = self.logmel_extractor(x)
        # print('specshape:',logmel_spec.shape)
        data['mel_spectrogram'] = logmel_spec.squeeze().transpose(0, 1).numpy()
        return data



class ToTensor(object):
    """Converts into a tensor."""

    def __init__(self, np_name, tensor_name, normalize=None):
        self.np_name = np_name
        self.tensor_name = tensor_name
        self.normalize = normalize

    def __call__(self, data):
        tensor = torch.FloatTensor(data[self.np_name])
        if self.normalize is not None:
            mean, std = self.normalize
            tensor -= mean
            tensor /= std
        data[self.tensor_name] = tensor
        return data
