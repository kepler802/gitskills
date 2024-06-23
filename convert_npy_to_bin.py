import numpy as np
import glob
import os
import sys
import tqdm
import librosa


def convert_file(srcf, dstf, sample_rate, seconds=3, louder_r=1):
    if not os.path.exists(srcf):
        print("not exist:", srcf)
        return
        
    if srcf.endswith(".npy"):
        data = np.load(srcf)
    elif srcf.endswith(".wav"):
        data, _ = librosa.load(srcf, sr = sample_rate, mono=False)
    if len(data.shape) == 1:
        data = data.reshape(1,len(data))

    # if "/collection_" in srcf:
    #     return
    data *= louder_r
    # print(data.shape, data[0], data.min(), data.max())
    # raise
    # if len(data.shape) == 2 and data.shape[0] == 2:
    #     print(data.shape)
    #     raise
    # print(srcf)
    if len(data.shape) == 2 and data.shape[0] == 2:
        data = (data[0,:] + data[1,:])*0.5
    # print(data.shape, data.min(), data.max(), "------", end="\t")
    # print('datashape:',data.shape)
    dst_len = int(sample_rate * seconds)
    audio = data.flatten()
    audio_length = len(audio)

    # 如果正样本音频小于0.5s，视为无效
    if "/0/" not in srcf and audio_length < 0.5 * sample_rate:
        print(audio_length)
        return None

    if audio_length < dst_len:
        start_point = dst_len - audio_length
        pad_data = np.zeros(dst_len, dtype=audio.dtype)
        # print(audio_length, pad_data.shape, start_point, start_point + audio_length)
        pad_data[start_point: start_point + audio_length] = audio
        audio = pad_data

    data = audio[-dst_len:]
    data = data*32767
    data = np.clip(data,-32768,32767)
    data = data.astype(np.int16)

    if len(audio[np.isnan(audio)]) > 0:
        print("has nan:", len(audio[np.isnan(audio)]))
        audio[np.isnan(audio)] = 0

    dstd = os.path.split(dstf)[0]
    os.makedirs(dstd,exist_ok=True)
    data.tofile(dstf)
    # print(data.shape, data.dtype, data.min(), data.max())
    return dstf


def append_path(str_dir):
    if str_dir[-1] != '/':
        str_dir = str_dir + '/'
    return str_dir


def get_convert_files(src_dir, dst_dir, sample_rate, seconds=3, louder_r=1):
    src_files = glob.glob(src_dir + '/**/*.wav', recursive=True)
    os.makedirs(dst_dir, exist_ok=True)
    file_list_path = dst_dir + '/files_list.txt'
    convert_files = []
    pbar = tqdm.tqdm(src_files)
    for srcf in pbar:
        bin_path = srcf.replace(src_dir, dst_dir)[:-4]
        bin_path = bin_path.replace(" ", "_").replace(".", "_")
        bin_path += '.bin'
        convertf = convert_file(srcf, bin_path, sample_rate, seconds, louder_r=louder_r)
        if convertf is not None:
            filename = convertf.replace(dst_dir + "/", '')
            convert_files.append(filename)

            # if len(convert_files) > 60:
            #     break
            
    with open(file_list_path, 'w') as f:
        for cf in convert_files:
            f.write(cf + '\n')
    

def scan_files_list(src_root):
    src_files = glob.glob(src_root+'/**/*.bin',recursive=True)
    convert_files = []
    pbar = tqdm.tqdm(src_files)
    for srcf in pbar:
        dstf = srcf.replace(src_root,'')
        convert_files.append(dstf)
    
    dstf = src_root + '/files_list.txt'
    with open(dstf,'w') as f:
        for cf in convert_files:
            f.write(cf+'\n')

        
def test_bin():
    # path = "/dataset/audio/audio_babycry/bin/test_cropped_sr16k_3s/1/301-Cryingbaby/Louise_01.m4a_0_0.bin"
    path = "/dataset/audio/audio_babycry/bin/test_cropped_sr16k_3s/-fPdOa99Iwg_30000_40000_sub_0.bin"
    audio = np.fromfile(path, dtype=np.int16)
    print(audio.shape, audio.dtype, audio.min(), audio.max())
    print(len(audio[np.isnan(audio)]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--sample_rate', type=int, default=8000)
    parser.add_argument('--seconds', type=int, default=2)
    parser.add_argument('--louder_r', type=int, default=1)
    args = parser.parse_args()

    get_convert_files(args.root, args.save_dir, args.sample_rate, args.seconds, args.louder_r)

    # scan_files_list("/dataset/audio/audio_command/task5_lns/eval")

    # test_bin()

"""
python convert_npy_to_bin.py \
--root /dataset/audio/audio_command/task5_lns/test4_5m_normsound_single_boda \
--save_dir /dataset/audio/audio_command/bin/task5_test4_5m_normsound_single_boda \
--sample_rate 8000 \
--seconds 2


python convert_npy_to_bin.py \
--root /dataset/audio/audio_babycry/test_cropped \
--save_dir /dataset/audio/audio_babycry/bin/test_cropped_sr8k_3s \
--sample_rate 16000 \
--seconds 3
"""