#!/usr/bin/env python
"""Train a CNN for Google speech commands."""

__author__ = 'Yuan Xu, Erdene-Ochir Tuguldur'

import argparse
import time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
from tqdm import *

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from datetime import datetime
import torchvision
from torchvision.transforms import *

from tensorboardX import SummaryWriter
import torchnet
import models
from datasets import *
from transforms import *
from mixup import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train-dataset", nargs='+',type=str, default='datasets/speech_commands/train', help='path of train dataset')
parser.add_argument("--valid-dataset", nargs='+',type=str, default='datasets/speech_commands/valid', help='path of validation dataset')
parser.add_argument("--background-noise", nargs='+',type=str, default='datasets/speech_commands/train/_background_noise_', help='path of background noise')
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--batch-size", type=int, default=128, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=6, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=1e-2, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
parser.add_argument("--learning-rate", type=float, default=1e-4, help='learning rate for optimization')
parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
parser.add_argument("--lr-scheduler-patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument("--lr-scheduler-step-size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.')
parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1, help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--max-epochs", type=int, default=70, help='max number of epochs')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument("--pretrain", type=str, help='checkpoint file to resume')
parser.add_argument("--model", choices=models.available_models, default=models.available_models[0], help='model of NN')
parser.add_argument("--input", type=int, default=32, help='input of NN')
parser.add_argument('--mixup', action='store_true', help='use mixup')
parser.add_argument("--save-path", type=str, help='checkpoint file to resume')
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if use_gpu:
    torch.backends.cudnn.benchmark = True

n_mels = args.input

save_path = args.save_path
print("save_path: ", save_path)
print("dataload_workers_nums: ", args.dataload_workers_nums)
os.makedirs(save_path, exist_ok=True)
data_aug_transform = Compose([ChangeAmplitude(), 
                              SplitAudio(time=2), 
                              ChangeSpeedPydub(max_scale=1.25), FixAudioLength(time=2), ToSTFT(n_fft=1024,hop_length=128), StretchAudioOnSTFT(max_scale=0.15), TimeshiftAudioOnSTFT(max_shift=4), FixSTFTDimension()])
bg_dataset = BackgroundNoiseDataset(args.background_noise, data_aug_transform, sample_rate=8000, sample_length = 2)
add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset,max_percentage=0.3)
train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels,  fixed_pcen=True), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
train_dataset = SpeechCommandsDataset(args.train_dataset,
                                Compose([LoadAudio(sample_rate=8000),
                                         data_aug_transform,
                                         add_bg_noise,
                                         train_feature_transform]))

valid_feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels,   fixed_pcen = True), ToTensor('mel_spectrogram', 'input')])
valid_dataset = SpeechCommandsDataset(args.valid_dataset,
                                Compose([LoadAudio(sample_rate=8000),
                                         FixAudioLength(time=2),
                                         valid_feature_transform]))

sample_num, weights = train_dataset.make_weights_for_balanced_classes()
sampler = WeightedRandomSampler(weights, sample_num)#len(weights))
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              pin_memory=False, num_workers=args.dataload_workers_nums)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=False, num_workers=args.dataload_workers_nums)


# a name used to save checkpoints etc.
full_name = '%s_%s_%s_bs%d_lr%.1e_wd%.1e' % (args.model, args.optim, args.lr_scheduler, args.batch_size, args.learning_rate, args.weight_decay)
if args.comment:
    full_name = '%s_%s' % (full_name, args.comment)

model = models.create_model(model_name=args.model, num_classes=len(CLASSES), in_channels=1)

if use_gpu:
    model = torch.nn.DataParallel(model).cuda()

criterion = torch.nn.CrossEntropyLoss()

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

start_timestamp = int(time.time()*1000)
start_epoch = 0
best_accuracy = 0
best_loss = 1e100
global_step = 0

if args.resume:
    print("resuming a checkpoint '%s'" % args.resume)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    model.float()
    optimizer.load_state_dict(checkpoint['optimizer'])

    best_accuracy = checkpoint.get('accuracy', best_accuracy)
    best_loss = checkpoint.get('loss', best_loss)
    start_epoch = checkpoint.get('epoch', start_epoch)
    global_step = checkpoint.get('step', global_step)

    del checkpoint  # reduce memory

if args.pretrain:
    print("load model from '%s'" % args.pretrain)
    pretrain_dict = torch.load(args.pretrain)['state_dict']
    model_dict = model.state_dict()

    for k in list(pretrain_dict.keys()):
        if pretrain_dict[k].shape != model_dict[k].shape:
            print("mismatch of {}, delect it".format(k))
            del pretrain_dict[k]

    model.load_state_dict(pretrain_dict, strict=False)
    model.float()
    # del checkpoint  # reduce memory

if args.lr_scheduler == 'plateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_scheduler_patience, factor=args.lr_scheduler_gamma)
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, last_epoch=start_epoch-1)

def get_lr():
    return optimizer.param_groups[0]['lr']

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter(logdir=os.path.join(save_path, current_time+"_" + full_name))

def train(epoch):
    global global_step
    confusion_matrix = torchnet.meter.ConfusionMeter(len(CLASSES))

    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
    phase = 'train'
    writer.add_scalar('%s/learning_rate' % phase,  get_lr(), epoch)

    model.train()  # Set model to training mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

        if args.mixup:
            inputs, targets = mixup(inputs, targets, num_classes=len(CLASSES))

        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward/backward
        outputs = model(inputs)
        if args.mixup:
            loss = mixup_cross_entropy_loss(outputs, targets)
        else:
            loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        if args.mixup:
            targets = batch['target']
            targets = Variable(targets, requires_grad=False).cuda()
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)
        confusion_matrix.add(pred.squeeze(), targets.data)

        writer.add_scalar('%s/loss' % phase, loss.item(), global_step)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    print(confusion_matrix.value())
    accuracy = correct/total
    epoch_loss = running_loss / it
    writer.add_scalar('%s/accuracy' % phase, 100*accuracy, epoch)
    writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

def valid(epoch, max_epochs):
    global best_accuracy, best_loss, global_step, save_path

    phase = 'valid'
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    confusion_matrix = torchnet.meter.ConfusionMeter(len(CLASSES))
    pbar = tqdm(valid_dataloader, unit="audios", unit_scale=valid_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

        inputs = Variable(inputs, volatile = True)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)
        confusion_matrix.add(pred.squeeze(), targets.data)

        writer.add_scalar('%s/loss' % phase, loss.item(), global_step)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    print(confusion_matrix.value())
    accuracy = correct/total
    epoch_loss = running_loss / it
    writer.add_scalar('%s/accuracy' % phase, 100*accuracy, epoch)
    writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    checkpoint = {
        'epoch': epoch,
        'step': global_step,
        'state_dict': model.state_dict(),
        'loss': epoch_loss,
        'accuracy': accuracy,
        'optimizer' : optimizer.state_dict(),
    }

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # if epoch > max_epochs//2:
        torch.save(checkpoint, save_path + '/best-acc-{:.3f}-{}-epoch{}.pth'.format(best_accuracy, full_name, epoch))
        torch.save(model, save_path + '/{}-{}-best-acc-{:.3f}-epoch{}.pth'.format(start_timestamp, full_name, best_accuracy, epoch))
    if epoch_loss < best_loss:
        best_loss = epoch_loss

        # torch.save(checkpoint, save_path + '/best-loss-{:.3f}-{}.pth'.format(best_loss, full_name))
        # torch.save(model, save_path + '/{}-{}-best-loss-{:.3f}.pth'.format(start_timestamp, full_name, best_loss))

    torch.save(checkpoint, save_path + '/last-speech-commands-checkpoint.pth')
    del checkpoint  # reduce memory

    return epoch_loss

print("training %s for Google speech commands..." % args.model)
since = time.time()
for epoch in range(start_epoch, args.max_epochs):
    if args.lr_scheduler == 'step':
        lr_scheduler.step()

    train(epoch)
    epoch_loss = valid(epoch, args.max_epochs)

    if args.lr_scheduler == 'plateau':
        lr_scheduler.step(metrics=epoch_loss)

    time_elapsed = time.time() - since
    time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
    print("%s, best accuracy: %.02f%%, best loss %f" % (time_str, 100*best_accuracy, best_loss))
print("finished")
