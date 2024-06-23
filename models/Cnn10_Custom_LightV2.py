
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torchaudio 

import numpy as np
import librosa


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=(2, 2)):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        self.avg_pool = nn.AvgPool2d(kernel_size=pool_size)
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_type='avg'):
        
        x = input
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = self.max_pool(x)
        elif pool_type == 'avg':
            x = self.avg_pool(x)
        elif pool_type == 'avg+max':
            x1 = self.avg_pool(x)
            x2 = self.max_pool(x)
            x = x1 + x2
        elif pool_type == 'keep':
            pass
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn10_Custom_LightV2(nn.Module):
    def __init__(self, num_classes, mel_bins=40, opt=None):
        
        super(Cnn10_Custom_LightV2, self).__init__()

        self.ref = 1.0
        self.amin = 1e-6
        self.top_db = None
        self.onnx_export = False
        self.fmin_aug_range = 5
        self.fmax_aug_range = 500
        self.mel_bins = mel_bins
        self.opt = opt
        self.mixup_lambda = None


        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=30, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        
        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=32, pool_size=(2, 2))
        self.conv_block2 = ConvBlock(in_channels=32, out_channels=64, pool_size=(2, 2))
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=128, pool_size=(2, 2))
        self.conv_block4 = ConvBlock(in_channels=128, out_channels=128, pool_size=(2, 2))

        if self.opt and self.opt.qat:
            self.pool = nn.AdaptiveAvgPool2d(output_size=(None, 1))

        #self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(128, num_classes, bias=True)
        
        self.init_weight()
        self.export_onnx = False

    def init_weight(self):
        init_bn(self.bn0)
        #init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, x):#64,1,63,40

        if  not self.export_onnx:
            x = x.transpose(2, 3)

        x = x.transpose(1, 3)
        # x = x.permute(0, 2, 3, 1)#64,40,63,1
        # print('xshape:',x.shape)
        x = self.bn0(x)
        x = x.transpose(1, 3)#64,1,63,40
        
        # if self.training: #and not self.opt.qat:
        #     #pass
        #     x = self.spec_augmenter(x)

        # Mixup on spectrogram
        # if self.training and self.mixup_lambda is not None and not self.opt.qat:
        #     x = do_mixup_ind(x, self.mixup_lambda)  ##没起作用
        
        x = self.conv_block1(x, pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)

        if self.opt and self.opt.qat:###没起作用
            x = self.pool(x)
            x = x[:, :, :, 0]
        else:
            x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        #x = F.dropout(x, p=0.2, training=self.training)
        #x = self.fc1(x)
        # print('fc1:',x)
        #x = F.relu_(x)
        # embedding = F.dropout(x, p=0.5, training=self.training)
        # clipwise_output = torch.sigmoid(self.fc_audioset(x))
        clipwise_output = self.fc_audioset(x)
        # print('res:',clipwise_output)
        # print('out:',clipwise_output)
        return clipwise_output