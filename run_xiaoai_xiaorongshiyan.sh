python train_speech_commands.py \
        --train-dataset  \
        /dataset/audio/fqk/zhuanlu1  \
        /dataset/audio/audio_command/task1_lns/yzx/train \
        /dataset/audio/audio_command/task1_lns/common_voice_normsound \
        /dataset/audio/audio_command/task1_lns/yzx/order \
        --valid-dataset  \
        /dataset/audio/audio_command/task1_lns/yzx/test/0.5m  \
        /dataset/audio/audio_command/task1_lns/yzx/test/1m  \
        /dataset/audio/audio_command/task1_lns/yzx/test/3m  \
        /dataset/audio/audio_command/task1_lns/yzx/test/5m  \
        --background-noise  \
        /dataset/audio/audio_command/collection_data/negative_sample_1226_linrui_cropped_normsound  \
        --pretrain \
        out/v23_add_pcen/best-acc-0.929-cnn10_sgd_plateau_bs256_lr1.0e-02_wd1.0e-02-epoch4.pth \
        --model                   cnn10  \
        --optim                   sgd  \
        --lr-scheduler            plateau  \
        --dataload-workers-nums   8 \
        --learning-rate           0.01  \
        --lr-scheduler-patience   10  \
        --max-epochs              100  \
        --batch-size              256  \
        --input                   40  \
        --save-path               out/xiaoai_fuxian_pcen_2

        # --pretrain \
        # out/v6/best-acc-0.959-vgg19_bn_sgd_plateau_bs256_lr1.0e-02_wd1.0e-02-epoch33.pth \

        # /dataset/audio/audio_command/task1_lns/train_zhvoice_normsound  \
        # /dataset/audio/audio_command/task1_lns/yzx/test/neg \

        # /dataset/audio/audio_command/task1_lns/yzx/train \
        # /dataset/audio/audio_command/task1_lns/common_voice_normsound \



        # /home/weiwei.dong/zhongtian_xiaoai/out/juntaida_16k_pretrained/best-acc-0.978-cnn10_sgd_plateau_bs128_lr1.0e-02_wd1.0e-02-epoch92.pth \
