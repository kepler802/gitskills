#!/bin/bash

# 设置 Python 环境变量和任何必需的路径（如果需要的话）
# export PYTHONPATH=/path/to/your/python/packages:$PYTHONPATH

# 运行 Python 程序
python /home/weiwei.dong/new_zhongtian/distillation.py \
    --train-dataset \
    /dataset/weiwei.dong/clean_addrir_addnoise_xiaoai \
    /dataset/audio/fqk/zhuanlu1 \
    /dataset/audio/audio_command/task1_lns/yzx/train \
    /dataset/audio/audio_command/task1_lns/common_voice_normsound \
    /dataset/audio/audio_command/task1_lns/yzx/order \
    --valid-dataset \
    /dataset/audio/audio_command/task1_lns/yzx/test/0.5m \
    /dataset/audio/audio_command/task1_lns/yzx/test/1m \
    /dataset/audio/audio_command/task1_lns/yzx/test/3m \
    /dataset/audio/audio_command/task1_lns/yzx/test/5m \
    --background-noise \
    /dataset/audio/audio_command/collection_data/negative_sample_1226_linrui_cropped_normsound \
    --pretrain \
    /home/weiwei.dong/ablation/checkpoint/v24_add_pcen_deploy_param_add_older/best-acc-0.894-cnn10_sgd_plateau_bs256_lr1.0e-02_wd1.0e-02-epoch37.pth \
    --model cnn10 \
    --optim sgd \
    --lr-scheduler plateau \
    --dataload-workers-nums 4 \
    --learning-rate 0.01 \
    --lr-scheduler-patience 10 \
    --max-epochs 100 \
    --batch-size 64 \
    --input 40 \
    --save-path /home/weiwei.dong/new_zhongtian/ablation/checkpoint/zhengliu_rir \
    --pt /home/weiwei.dong/Wav2Keyword/wav2vec_small.pt \
    # --ptmy /dataset/SpeechCommands/train_model_zhuanlu16k_b/best_model.pth
    # /home/zhenxing.ye/work/pytorch-speech-commands/out/v23_add_pcen/best-acc-0.929-cnn10_sgd_plateau_bs256_lr1.0e-02_wd1.0e-02-epoch4.pth \
    # --pretrain \
    # /home/weiwei.dong/zhongtian_xiaoai/out/v24_add_pcen_deploy_param_add_older_303masknone/best-acc-0.966-cnn10_sgd_plateau_bs256_lr1.0e-02_wd1.0e-02-epoch31.pth \
