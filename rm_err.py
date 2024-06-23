
import os
import shutil

err_path = "/home/zhenxing.ye/work/pytorch-speech-commands/out/shiyun_v3/err/Mar21_13-59-05.txt"
right_list = [
    "train_older_normsound#1#older_nihaoshiyun#nihaoshiyun_common_voice_fr_19140966.wav",
    "train_older_normsound#1#older_nihaoshiyun#nihaoshiyun_common_voice_fr_19537525.wav",
    "train_older_normsound#1#older_nihaoshiyun#nihaoshiyun_common_voice_fr_19648059.wav",
    "train_older_normsound#1#older_nihaoshiyun#nihaoshiyun_common_voice_fr_19701802.wav",
    "train_older_normsound#1#older_nihaoshiyun#nihaoshiyun_common_voice_fr_19714814.wav",
    "train_older_normsound#1#older_nihaoshiyun#nihaoshiyun_common_voice_fr_19794962.wav",
    "train_older_normsound#1#older_nihaoshiyun#nihaoshiyun_common_voice_fr_19954233.wav",
    "train_older_normsound#1#older_nihaoshiyun#nihaoshiyun_common_voice_fr_20256265.wav",
    "train_older_normsound#2#older_bodadianhua#bodadianhua_common_voice_fr_20659115.wav",
    "train_older_normsound#2#older_bodadianhua#bodadianhua_common_voice_fr_22450861.wav",
    "train_older_normsound#2#older_bodadianhua#bodadianhua_common_voice_fr_22955008.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_17340737.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_17759759.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_17872603.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_18154485.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_18704975.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_19148332.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_19446040.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_19598938.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_19627763.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_19656382.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_19680865.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_19712854.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_19784337.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_19796389.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_19962596.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_20041148.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_20509258.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_20514101.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_20517317.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_22173236.wav",
    "train_older_normsound#3#older_guanbipingmu#guanbipingmu_common_voice_fr_22982619.wav",
    "train_older_normsound#4#older_dakaipingmu#dakaipingmu_common_voice_fr_18513025.wav",
    "train_older_normsound#4#older_dakaipingmu#dakaipingmu_common_voice_fr_19598938.wav",
    "train_older_normsound#4#older_dakaipingmu#dakaipingmu_common_voice_fr_19716942.wav",
    "train_older_normsound#4#older_dakaipingmu#dakaipingmu_common_voice_fr_19751472.wav",
    "train_older_normsound#4#older_dakaipingmu#dakaipingmu_common_voice_fr_19874092.wav",
    "train_older_normsound#4#older_dakaipingmu#dakaipingmu_common_voice_fr_19954233.wav",
    "train_older_normsound#4#older_dakaipingmu#dakaipingmu_common_voice_fr_20533906.wav",
    "train_older_normsound#4#older_dakaipingmu#dakaipingmu_common_voice_fr_21872921.wav",
    "train_older_normsound#4#older_dakaipingmu#dakaipingmu_common_voice_fr_22322479.wav",
]



with open(err_path, "r") as f:
    lines = f.read().splitlines()

    for line in lines[1:-4]: #第一个误删了，最后4个是0类

        f_path , cls = line.split(" ")

        f_name = "#".join(f_path.split("/")[-4:])
        cls = int(line.split(" ")[-1])

        if f_name not in right_list:

            dst_path = f_path.replace("train_older_normsound", "train_older_normsound_err")

            os.makedirs(os.path.dirname(dst_path), exist_ok= True)

            shutil.move(f_path, dst_path)

