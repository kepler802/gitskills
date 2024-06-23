import glob
import random

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
from tqdm import *

from torch.utils.data import DataLoader
from torchvision.transforms import *
from datasets import *
from transforms import *


data_dict = {

"/dataset/audio/audio_command/task_hi_smart/article_train_bg": [10, -1],
"/dataset/audio/audio_command/task_hi_smart/office_bg/train": [16, -1],
"/dataset/audio/audio_command/task_hi_smart/ln_old_bg"   : [8, -1],

"/dataset/audio/audio_command/task_hi_smart/gen_TTS/inside_3s_8k": [-1, 8],
"/dataset/audio/audio_command/task_hi_smart/gen_TTS_zhuanlu/1m"  : [-1, 8],
"/dataset/audio/audio_command/task_hi_smart/gen_TTS_zhuanlu/3m"  : [-1, 8],
"/dataset/audio/audio_command/task_hi_smart/gen_TTS_zhuanlu/5m" : [-1, 8],
"/dataset/audio/audio_command/task_hi_smart/yzx/RTVC_zhuanlu_val/1m"  : [-1, 8],
"/dataset/audio/audio_command/task_hi_smart/yzx/RTVC_zhuanlu_val/3m" : [-1, 8],
"/dataset/audio/audio_command/task_hi_smart/yzx/RTVC_zhuanlu_val/5m" : [-1, 8],

"/dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/1m": [-1, 20],
"/dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/3m": [-1, 20],
"/dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/5m": [-1, 20],

}


def gen_quant_file(info_dict):

    dst_dict = {}

    for k, v in info_dict.items():
        dir_list = os.listdir(k)

        for dir in dir_list:

            dir_path = os.path.join(k, dir)

            if int(dir) == 0:
                assert v[0] > 0
                dst_num = v[0]
            else:
                assert v[1] > 0
                dst_num = v[1]//(len(dir_list) - 1) if "0" in dir_list else v[1]//len(dir_list)

            data_list = glob.glob(dir_path + "/**/*.wav", recursive=True)
            random.shuffle(data_list)
            for f in data_list[:dst_num]:
                dst_dict[f] = int(dir)
    return dst_dict

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size", type=int, default=64, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--input", choices=['mel32', 'mel40'], default='mel40', help='input of NN')
parser.add_argument('--output', type=str, default='', help='save output to file for the kaggle competition, if empty the model name will be used')
args = parser.parse_args()


n_mels = 32
if args.input == 'mel40':
    n_mels = 40

output_dir = args.output

dst_dict = gen_quant_file(data_dict)


feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels, fixed_pcen=True), ToTensor('mel_spectrogram', 'input')])
transform = Compose([LoadAudio(sample_rate=8000), FixAudioLength(time=2), feature_transform])

test_dataset = SpeechCommandsDataset(dst_dict, transform,)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=None,
                            pin_memory=True, num_workers=args.dataload_workers_nums)

pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)
for batch in pbar:
    inputs = batch['input']
    inputs = torch.unsqueeze(inputs, 1)
    inputs = inputs.transpose(2, 3)

    inputs = inputs.numpy()

    f_names = batch['path']

    for i in range(len(f_names)):
        arr = inputs[i:i+1, ...]

        save_path = os.path.join(output_dir, f_names[i].split("/")[-1][:-4] + ".npy")

        np.save(save_path, arr)

