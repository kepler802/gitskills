#!/usr/bin/env python
"""Test a pretrained CNN for Google speech commands."""

__author__ = 'Yuan Xu, Erdene-Ochir Tuguldur'

import argparse
import time
import csv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import *
import torchnet
import models
from datasets import *
from transforms import *
from datetime import datetime

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("model", help='a pretrained neural network model')
parser.add_argument("model_name", help='model_name')
parser.add_argument("--dataset-dir", nargs='+',type=str, default='datasets/speech_commands/test', help='path of test dataset')
parser.add_argument("--background-noise", nargs='+',type=str, default='datasets/speech_commands/train/_background_noise_', help='path of background noise')
parser.add_argument("--batch-size", type=int, default=128, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--input", choices=['mel32', 'mel40'], default='mel32', help='input of NN')
parser.add_argument('--multi-crop', action='store_true', help='apply crop and average the results')
parser.add_argument('--generate-kaggle-submission', action='store_true', help='generate kaggle submission file')
parser.add_argument("--kaggle-dataset-dir", type=str, default='datasets/speech_commands/kaggle', help='path of kaggle test dataset')
parser.add_argument('--output', type=str, default='', help='save output to file for the kaggle competition, if empty the model name will be used')
#parser.add_argument('--prob-output', type=str, help='save probabilities to file', default='probabilities.json')
args = parser.parse_args()

dataset_dir = args.dataset_dir
if args.generate_kaggle_submission:
    dataset_dir = args.kaggle_dataset_dir

print("loading model...")
model = torch.load(args.model)
if type(model) == dict:
    real_model = models.create_model(model_name=args.model_name, num_classes=len(CLASSES), in_channels=1)
    real_model = torch.nn.DataParallel(real_model).cuda()
    real_model.load_state_dict(model['state_dict'])
    model = real_model

model.float()

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if use_gpu:
    torch.backends.cudnn.benchmark = True
    model.cuda()

n_mels = 32
if args.input == 'mel40':
    n_mels = 40

feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels, fixed_pcen=True), ToTensor('mel_spectrogram', 'input')])
# feature_transform = Compose([ExtractFeature(sample_rate=8000), ToTensor('mel_spectrogram', 'input')])
transform = Compose([LoadAudio(sample_rate=8000), FixAudioLength(time=2), feature_transform])





# transform = Compose([LoadAudio(sample_rate=22050), FixAudioLengthTail(time=2), feature_transform])

# test_dataset = SpeechCommandsDataset(dataset_dir, transform,)#silence_percentage=0)
# test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=None,
#                             pin_memory=use_gpu, num_workers=args.dataload_workers_nums)


# data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(time=2), ToSTFT(n_fft=1024,hop_length=128), StretchAudioOnSTFT(max_scale=0.15), TimeshiftAudioOnSTFT(max_shift=4), FixSTFTDimension()])
# bg_dataset = BackgroundNoiseDataset(args.background_noise, data_aug_transform, sample_rate=8000, sample_length = 2)
# add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset,max_percentage=0.3)
# train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
# train_dataset = SpeechCommandsDataset(dataset_dir,
#                                 Compose([LoadAudio(sample_rate=8000),
#                                          data_aug_transform,
#                                          add_bg_noise,
#                                          train_feature_transform]))
# # weights = train_dataset.make_weights_for_balanced_classes()
# # sampler = WeightedRandomSampler(weights, len(weights))
# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
#                               pin_memory=False, num_workers=args.dataload_workers_nums)

# test_dataloader = train_dataloader

criterion = torch.nn.CrossEntropyLoss()

def multi_crop(inputs):
    b = 1
    size = inputs.size(3) - b * 2
    patches = [inputs[:, :, :, i*b:size+i*b] for i in range(3)]
    outputs = torch.stack(patches)
    outputs = outputs.view(-1, inputs.size(1), inputs.size(2), size)
    outputs = torch.nn.functional.pad(outputs, (b, b, 0, 0), mode='replicate')
    return torch.cat((inputs, outputs.data))

def test():
    model.eval()  # Set model to evaluate mode

    #running_loss = 0.0
    #it = 0
    correct_global = 0
    total_total = 0
    err_path = []
    err_pred = []

    for i in range(1): ##
        test_dataset = SpeechCommandsDataset(dataset_dir, transform,)

    # for dir in dataset_dir:
    #     test_dataset = SpeechCommandsDataset([dir], transform,)


        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=None,
                                    pin_memory=use_gpu, num_workers=args.dataload_workers_nums)

        correct, total = 0, 0
        confusion_matrix = torchnet.meter.ConfusionMeter(len(CLASSES))

        pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)
        for batch in pbar:
            inputs = batch['input']
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch['target']

            n = inputs.size(0)
            if args.multi_crop:
                inputs = multi_crop(inputs)

            inputs = Variable(inputs, volatile = True)
            targets = Variable(targets, requires_grad=False)

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # forward
            outputs = model(inputs)
            #loss = criterion(outputs, targets)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            if args.multi_crop:
                outputs = outputs.view(-1, n, outputs.size(1))
                outputs = torch.mean(outputs, dim=0)
                outputs = torch.nn.functional.softmax(outputs, dim=1)

            # statistics
            #it += 1
            #running_loss += loss.data[0]
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()
            total += targets.size(0)

            confusion_matrix.add(pred.squeeze(-1), targets.data)

            filenames = batch['path']

            err_idx = torch.where(pred.squeeze()!=targets)[0]
            for ids in err_idx.tolist():
                err_path.append(filenames[ids])
                err_pred.append(pred[ids].item())

        accuracy = correct/total
        #epoch_loss = running_loss / it
        correct_global += correct
        total_total += total

        matrix_arr = confusion_matrix.value()

        err = np.sum(matrix_arr[0, 1:]) / np.sum(matrix_arr)
        recall = np.sum([matrix_arr[i][i] for i in range(-(matrix_arr.shape[0]-1),0)]) / np.sum(matrix_arr[-(matrix_arr.shape[0]-1):, :])

        print("准确率/检出率/误判率:")
        print("{:.3f}/{:.3f}/{:.3f}".format(accuracy, recall, err))
        print("confusion matrix:")
        print(matrix_arr)

    print("整体准确率：{:.3f}".format(correct_global/total_total))
    return err_path, err_pred

print("testing...")
err_path, err_pred = test()
exit()
save_dir = os.path.join(os.path.dirname(args.model), "err")
save_dir ="hi_smarts/err"
os.makedirs(save_dir, exist_ok=True)
txt_name = datetime.now().strftime('%b%d_%H-%M-%S') + ".txt"


with open(os.path.join(save_dir, txt_name), "w") as f:
    for i in range(len(err_path)):
        f.write("{} {}\n".format(err_path[i], err_pred[i]))

for i in range(-(matrix_arr.shape[0]),0):
    print(i)