python train_speech_commands.py \
        --train-dataset  \
        /dataset/audio/audio_command/task5_lns/yzx/mix  \
        /dataset/audio/audio_command/task5_lns/yzx/zhuanlu_ori \
        /dataset/audio/audio_command/task5_lns/train_zhvoice_normsound \
        /dataset/audio/audio_command/task5_lns/yzx/common_voice \
        --valid-dataset  \
        /dataset/audio/audio_command/task5_lns/yzx/test/test_normsound  \
        /dataset/audio/audio_command/task5_lns/yzx/test/1m_normsound  \
        /dataset/audio/audio_command/task5_lns/yzx/test/3m_normsound \
        /dataset/audio/audio_command/task5_lns/yzx/test/5m_normsound  \
        --background-noise  \
        /dataset/audio/audio_command/collection_data/negative_sample_1226_linrui_cropped_normsound  \
        --pretrain \
        out/shiyun_v4/best-acc-0.928-cnn10_sgd_plateau_bs256_lr1.0e-03_wd1.0e-02-epoch1.pth \
        --model                   cnn10  \
        --optim                   sgd  \
        --lr-scheduler            plateau  \
        --dataload-workers-nums   4 \
        --learning-rate           0.01  \
        --lr-scheduler-patience   10  \
        --max-epochs              100  \
        --batch-size              256  \
        --input                   40  \
        --save-path               out/shiyun_v7

        # --pretrain \
        # out/v6/best-acc-0.959-vgg19_bn_sgd_plateau_bs256_lr1.0e-02_wd1.0e-02-epoch33.pth \

        # /dataset/audio/audio_command/task1_lns/train_zhvoice_normsound  \
        # /dataset/audio/audio_command/task1_lns/yzx/test/neg \

        # /dataset/audio/audio_command/task1_lns/yzx/train \
        # /dataset/audio/audio_command/task1_lns/common_voice_normsound \