"""Google speech commands dataset."""
__author__ = 'Yuan Xu'

import os
import numpy as np

import librosa

from torch.utils.data import Dataset
import glob
from tqdm import tqdm
import random

__all__ = [ 'CLASSES', 'SpeechCommandsDataset', 'BackgroundNoiseDataset' ]

# CLASSES = '0, 1, 2, 3'.split(', ')
CLASSES = '0, 1, 2, 3, 4'.split(', ')
# CLASSES = 'unknown, yes, no, up, down, left, right, on, off, stop, go'.split(', ')
class SpeechCommandsDataset(Dataset):
    """Google speech commands dataset. Only 'yes', 'no', 'up', 'down', 'left',
    'right', 'on', 'off', 'stop' and 'go' are treated as known classes.
    All other classes are used as 'unknown' samples.
    See for more information: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
    """

    def __init__(self, folder_list, transform=None, classes=CLASSES, silence_percentage=0.1):

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        data = []

        ignore_path = "/dataset/audio/audio_command/task1_lns/yzx/test_neg.txt"

        with open(ignore_path, "r") as f:
            ignore_lines = f.read().splitlines()

        if type(folder_list) == dict:
            for k, v in folder_list.items():
                data.append((k, v))
        
        else:

            for folder in folder_list:

                all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
            #for c in classes[2:]:
            #    assert c in all_classes

                for c in all_classes:
                    if c not in class_to_idx:
                        class_to_idx[c] = 0
                    d = os.path.join(folder, c)
                    target = class_to_idx[c]

                    f_list = glob.glob(d + '/**/*.wav', recursive=True)

                    for f in f_list:

                        f_info = "#".join(f.split("/")[-2:])
                        if f_info in ignore_lines:
                            continue

                        # if target == 0 and f.split("/")[-2] in ["1m", "3m", "5m"] and random.random() < 0.5:
                        #     continue 

                        data.append((f, target))

        # add silence
        # target = class_to_idx['silence']
        # data += [('', target)] * int(len(data) * silence_percentage)

        self.classes = classes
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)
            value = data.get('input2', None)  # 如果不存在键 'input'，则返回None

            if value is not None:
                data =  {"input": data['input'],"source": data['input2'],"target": data['target'],'path': data['path']}
                # 在这里处理value
            else:
                data =  {"input": data['input'],"target": data['target'],'path': data['path']}

        return data

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        sample_num = int(np.sum(count[1:]) * nclasses / (nclasses - 1))
        N = float(sum(count))
        weight_per_class = N / count
        # weight_per_class[0] *= 2
        # weight_per_class[0] *= 4
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return sample_num, weight

class BackgroundNoiseDataset(Dataset):
    """Dataset for silence / background noise."""

    def __init__(self, folder_list, transform=None, sample_rate=16000, sample_length=1):

        samples = []

        ignore_path = "/dataset/audio/audio_command/task1_lns/yzx/test_neg.txt"

        with open(ignore_path, "r") as f:
            ignore_lines = f.read().splitlines()

        for folder in folder_list:
            f_list = glob.glob(folder + '/**/*.wav', recursive=True)

            for f in tqdm(f_list):
                f_info = "#".join(f.split("/")[-2:])
                if f_info in ignore_lines:
                    continue

            # for f in tqdm(f_list[:100]):
                s, sr = librosa.load(f, sample_rate)
                samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.classes = CLASSES
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data
