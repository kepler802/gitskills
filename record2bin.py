import os
import torch
import torchaudio
import heapq
import numpy as np
from tqdm import tqdm

ori_sr = 16000
bin_sr = 8000



def pcm2audio(pcm_file, num_channels = 1, sample_width = 2 ):

    num_samples = os.path.getsize(pcm_file) // (num_channels * sample_width)

    # 读取PCM文件
    pcm_tensor = torch.from_file(pcm_file, dtype=torch.int16, size=num_samples)
    pcm_tensor = pcm_tensor.float() / (2**(sample_width * 8 - 1))  # 归一化到[-1.0, 1.0]

    # 重塑张量以匹配声道数
    pcm_tensor = pcm_tensor.view(1, -1)

    return pcm_tensor    

def NormSound(audio_data, max_rate=0.3, top_num=50):

    if top_num == 0:
        return audio_data
    
    audio_abs = abs(audio_data[0])
    top_data = heapq.nlargest(top_num, audio_abs)
    top_mean = sum(top_data) / top_num
    r = max_rate / top_mean if top_mean != 0 else 1

    return audio_data * r

def load_audio(f_path):

    sr = ori_sr

    if f_path.endswith(".wav"):
        audio, sr = torchaudio.load(f_path)
    else:
        audio = pcm2audio(f_path)
    
    return audio, sr


input_dir = "/dataset/audio/audio_command/collection_sequence/1m"

# output_dir = "/dataset/audio/audio_command/task1_lns/realtime_test"
output_dir = "real_time_bin"

for f in tqdm(os.listdir(input_dir)):
    input_path = os.path.join(input_dir, f)

    data, ori_sr = load_audio(input_path)


    if ori_sr != bin_sr:
        data = torchaudio.transforms.Resample(orig_freq = ori_sr, new_freq = bin_sr)(data)

    data = NormSound(data).numpy()
    data = data*32767
    data = np.clip(data,-32768,32767)
    data = data.astype(np.int16)

    out_name = "#".join([input_dir.split("/")[-1], f[:-3]+"bin"])
    output_bin = os.path.join(output_dir, out_name)

    data.tofile(output_bin)



