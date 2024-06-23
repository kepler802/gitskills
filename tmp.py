# import scipy.io as scio
# from scipy.io import wavfile
import numpy as np
# import torchaudio
# import torch
# import os
# from pydub import AudioSegment

# from scipy.io.wavfile import read
# import glob
# from tqdm import tqdm



aaa = np.load("/home/qing.xiang/models_quant/sound_recognition/audio_task1_mix_2s_sr8k_v25/dakaipingmu_command_11_office.npy")





# audio=AudioSegment.from_wav("aaa.wav")

# aaa = audio.get_array_of_samples()
# # sample_rate, wav_sample = scipy.io.wavfile.read("aaa.wav") 

# wav_sample = wav_sample[:, None]
# aaa = audiosegment.from_numpy_array(wav_sample, sample_rate)
# aaa.export("ssss.wav",  format = "wav")
# segment = AudioSegment(data=wav_sample.tobytes(),
#                        sample_width=4,
#                        frame_rate=sample_rate, channels=1)

# segment.export("ssss.wav",  format = "wav")

def test_speed(audio_path, save_dir):
        

        samples, sr = torchaudio.load(audio_path)
        samples = samples.numpy().squeeze()
        # scale = random.uniform(-0.3, 0)
        speed_fac = 1.45
        samples = np.interp(np.arange(0, len(samples), speed_fac), np.arange(0,len(samples)), samples).astype(np.float32)    

        samples = torch.from_numpy(samples[None, :])
        save_path = os.path.join(save_dir, "{:.2f}_{}".format(speed_fac, audio_path.split("/")[-1]))

        torchaudio.save(save_path, samples, sr)





def RandomSpeedChange(audio_path, save_dir, min_rate=0.8, max_rate=1.5):

        audio_data, sr = torchaudio.load(audio_path)

        # speed = random.uniform(min_rate, max_rate)
        speed = 1.6


        spectrogram = torchaudio.functional.spectrogram(
                audio_data, n_fft=1024, win_length=1024, hop_length=256, power=None, pad=0, window=None, normalized=False,
        )

        # 使用torchaudio对频谱张量进行Time-Scale Modification
        stretcher = torchaudio.transforms.TimeStretch(
                hop_length=256, n_freq=513, fixed_rate=speed
        )
        stretched_spectrogram = stretcher(spectrogram)

        # 将频谱张量转换为音频信号
        audio_data = torch.istft(
                stretched_spectrogram, n_fft=1024, win_length=1024, hop_length=256,
        )

        save_path = os.path.join(save_dir, "{:.2f}_{}".format(speed, audio_path.split("/")[-1]))
        torchaudio.save(save_path, audio_data, sr)

def pydub_speed(audio_path, save_dir,):
        
        speed = 1.6
        sample_width = 2

        audio_data, sr = torchaudio.load(audio_path)

        audio_data = audio_data[0].numpy()

        audio_data = np.array(audio_data*(2**(sample_width * 8 - 1)), dtype=np.int16)


        # byte_io = io.BytesIO(audio_data.tobytes())

        audio = AudioSegment(audio_data.tobytes(), sample_width=sample_width, frame_rate = sr, channels = 1)
        # audio.export("ssss.wav",  format = "wav")

        # audio = AudioSegment.from_raw(byte_io, sample_width=2, frame_rate = sr, channels = 1)
        fast_audio = audio.speedup(playback_speed = speed)

        save_path = os.path.join(save_dir, "{:.2f}_{}".format(speed, audio_path.split("/")[-1]))

        fast_audio.export(save_path,  format = "wav")




def location_verify(input_dir):
        file_list = glob.glob(input_dir + "/**/*.wav", recursive=True)

        count = 0
        for f in tqdm(file_list):
                split_info = f.split("/")[-1][:-4].split("#")[-1].split("_")
                split_info = [float(i) for i in split_info]

                if  split_info[1] - split_info[0] < 0.1 or   \
                    split_info[3] - split_info[2] < 0.1 or split_info[5] - split_info[4] < 0.1 or split_info[7] - split_info[6] < 0.1:
                     count += 1

                     print(f)   

                # if  split_info[-1] - split_info[0] < 0.2:
                #         print(f)                                

        print(count, count/len(file_list))

input_dir = "/dataset/audio/audio_command/task1_lns/common_voice_normsound"
location_verify(input_dir)

# split_info[-1] - split_info[0] < 0.9 or












audio_path = "/dataset/audio/audio_command/task1_lns/train_zhvoice_normsound/0/zhthchs30/zhthchs30#A14#A14_98.wav"
# audio_path = "aaa.wav"
save_dir = "tmp_test/test_speed_pydub"
pydub_speed(audio_path, save_dir,)


RandomSpeedChange(audio_path, save_dir)












# audio_path = "tmp_test/v16_test_err/0#1m_dakaiqianlu#dakaiqianlu_command_8_office.wav"
audio_path = "aaa.wav"
save_dir = "tmp_test/test_speed"
test_speed(audio_path, save_dir)


file = "babble.mat"

data = scio.loadmat(file)
data = data["babble"]

wavfile.write( file[:-4] + '.wav', 19980, data)

aa=0