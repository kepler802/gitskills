import torchaudio




audio_path = "/home/zhenxing.ye/work/pytorch-speech-commands/ln_data/hi_smart_train/train_class0/0/1m#0865.wav"


audio, sr = torchaudio.load(audio_path)



audio_path ="/home/xueting.ma/语音数据的处理/自己采集的音频/组员录音_分割后标准化wav文件_未cover_重命名_标准化/5m/3/shut_down_4.wav"

audio2, sr2 = torchaudio.load(audio_path)


aaa = 0