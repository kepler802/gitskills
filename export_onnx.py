import argparse
import time
import csv
import os

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
parser.add_argument('--onnx_path', type=str, default='', help='path to export onnx')
args = parser.parse_args()


if args.onnx_path == '':
    args.onnx_path = os.path.join(os.path.dirname(args.model), args.model.split("/")[1] + ".onnx")


print("loading model...")
model = torch.load(args.model)
if type(model) == dict:
    real_model = models.create_model(model_name=args.model_name, num_classes=len(CLASSES), in_channels=1)
    real_model = torch.nn.DataParallel(real_model)
    real_model.load_state_dict(model['state_dict'])
    model = real_model

model.float()
model.eval() 
model.module.export_onnx = True

dummy_input = torch.rand(1, 1, 126, 40,
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device)

torch.onnx.export(model.module, dummy_input, 
                  args.onnx_path, 
                  verbose=True,
                  input_names=['x'],
                  output_names=['cls'],
                  opset_version=11)


print('export success')