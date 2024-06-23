CUDA_VISIBLE_DEVICES=0,1 \
python train_speech_commands.py \
        --train-dataset  \
        /dataset/audio/audio_command/task_hi_smart/office_bg/train \
        /dataset/audio/audio_command/task_hi_smart/article_train_bg  \
        /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/1m \
        /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/3m \
        /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/5m \
        --valid-dataset  \
        /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/1m \
        /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/3m \
        /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/5m \
        --background-noise  \
        /dataset/audio/audio_command/task_hi_smart/article_train_bg  \
        --pretrain \
        out/hi_smart_v10/best-acc-0.970-cnn10_sgd_plateau_bs256_lr1.0e-03_wd1.0e-02-epoch6.pth \
        --model                   cnn10  \
        --optim                   sgd  \
        --lr-scheduler            plateau  \
        --dataload-workers-nums   8 \
        --learning-rate           0.0001  \
        --lr-scheduler-patience   10  \
        --max-epochs              20  \
        --batch-size              64  \
        --input                   40  \
        --save-path               out/hi_smart_v12

        # --train-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/office_bg/train \
        # /dataset/audio/audio_command/task_hi_smart/yzx/article_zhuanlu_train/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/article_zhuanlu_train/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/article_zhuanlu_train/5m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/5m \
        # --valid-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/5m \
        # --background-noise  \
        # /dataset/audio/audio_command/task_hi_smart/article_train_bg  \
        # --pretrain \
        # out/hi_smart_v10/best-acc-0.970-cnn10_sgd_plateau_bs256_lr1.0e-03_wd1.0e-02-epoch6.pth \
        # --model                   cnn10  \
        # --optim                   sgd  \
        # --lr-scheduler            plateau  \
        # --dataload-workers-nums   8 \
        # --learning-rate           0.0001  \
        # --lr-scheduler-patience   10  \
        # --max-epochs              20  \
        # --batch-size              64  \
        # --input                   40  \
        # --save-path               out/hi_smart_v11

        # --train-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/article_train_bg \
        # /dataset/audio/audio_command/task_hi_smart/office_bg/train \
        # /dataset/audio/audio_command/task_hi_smart/ln_old_bg  \
        # /dataset/audio/audio_command/task_hi_smart/yzx/article_zhuanlu_train/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/article_zhuanlu_train/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/article_zhuanlu_train/5m \
        # /dataset/audio/audio_command/task_hi_smart/gen_TTS/inside_3s_8k \
        # /dataset/audio/audio_command/task_hi_smart/gen_TTS_zhuanlu/1m \
        # /dataset/audio/audio_command/task_hi_smart/gen_TTS_zhuanlu/3m \
        # /dataset/audio/audio_command/task_hi_smart/gen_TTS_zhuanlu/5m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/RTVC_zhuanlu_train/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/RTVC_zhuanlu_train/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/RTVC_zhuanlu_train/5m \
        # --valid-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/office_bg/test \
        # /dataset/audio/audio_command/task_hi_smart/yzx/article_zhuanlu_val/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/article_zhuanlu_val/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/article_zhuanlu_val/5m \
        # /dataset/audio/audio_command/task_hi_smart/gen_RTVC \
        # /dataset/audio/audio_command/task_hi_smart/yzx/RTVC_zhuanlu_val/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/RTVC_zhuanlu_val/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/RTVC_zhuanlu_val/5m \
        # --background-noise  \
        # /dataset/audio/audio_command/collection_data/negative_sample_1226_linrui_cropped_normsound  \
        # --pretrain \
        # out/hi_smart_v8/best-acc-0.949-cnn10_sgd_plateau_bs256_lr1.0e-03_wd1.0e-02-epoch11.pth \
        # --model                   cnn10  \
        # --optim                   sgd  \
        # --lr-scheduler            plateau  \
        # --dataload-workers-nums   8 \
        # --learning-rate           0.001  \
        # --lr-scheduler-patience   5  \
        # --max-epochs              50  \
        # --batch-size              256  \
        # --input                   40  \
        # --save-path               out/hi_smart_v10

        # --train-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/train_class0 \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/5m \
        # --valid-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/5m \
        # --background-noise  \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_train  \
        # --pretrain \
        # out/hi_smart_v8/best-acc-0.949-cnn10_sgd_plateau_bs256_lr1.0e-03_wd1.0e-02-epoch11.pth \
        # --model                   cnn10  \
        # --optim                   sgd  \
        # --lr-scheduler            plateau  \
        # --dataload-workers-nums   8 \
        # --learning-rate           0.0001  \
        # --lr-scheduler-patience   10  \
        # --max-epochs              20  \
        # --batch-size              64  \
        # --input                   40  \
        # --save-path               out/hi_smart_v9

        # --train-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_train \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/train_class0 \
        # /dataset/audio/audio_command/task_hi_smart/speech_commands_neg  \
        # /dataset/audio/audio_command/task_hi_smart/gen_commands_pad/inside_3s_8k \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/1m \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/3m \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/5m \
        # /home/xueting.ma/Voice_Processing/自己采集的音频/代码生成的val指令_转录后分割/3m \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_8k/3m  \
        # --valid-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/speech_command_val_8k \
        # /home/xueting.ma/Voice_Processing/自己采集的音频/代码生成的val指令_转录后分割/1m \
        # /home/xueting.ma/Voice_Processing/自己采集的音频/代码生成的val指令_转录后分割/5m \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/test \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_8k/1m  \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_8k/5m  \
        # --background-noise  \
        # /dataset/audio/audio_command/collection_data/negative_sample_1226_linrui_cropped_normsound  \
        # --pretrain \
        # out/hi_smart_v7/best-acc-0.873-cnn10_sgd_plateau_bs1024_lr1.0e-03_wd1.0e-02-epoch3.pth \
        # --model                   cnn10  \
        # --optim                   sgd  \
        # --lr-scheduler            plateau  \
        # --dataload-workers-nums   8 \
        # --learning-rate           0.001  \
        # --lr-scheduler-patience   5  \
        # --max-epochs              50  \
        # --batch-size              256  \
        # --input                   40  \
        # --save-path               out/hi_smart_v8

        # --train-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/gen_commands_pad/inside_3s_8k \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_train \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/train_class0 \
        # /dataset/audio/audio_command/task_hi_smart/speech_commands_neg  \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/1m \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/3m \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/5m \
        # --valid-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/speech_command_val_8k \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/1m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/3m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/5m \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/test \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_8k/1m  \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_8k/3m  \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_8k/5m  \
        # --background-noise  \
        # /dataset/audio/audio_command/collection_data/negative_sample_1226_linrui_cropped_normsound  \
        # --pretrain \
        # /home/zhenxing.ye/work/pytorch-speech-commands/checkpoints/best-acc-0.867-cnn10_sgd_plateau_bs1024_lr1.0e-02_wd1.0e-02-epoch17.pth \
        # --model                   cnn10  \
        # --optim                   sgd  \
        # --lr-scheduler            plateau  \
        # --dataload-workers-nums   8 \
        # --learning-rate           0.001  \
        # --lr-scheduler-patience   10  \
        # --max-epochs              50  \
        # --batch-size              1024  \
        # --input                   40  \
        # --save-path               out/hi_smart_v7

        # --train-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/train_class0 \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/5m \
        # --valid-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/5m \
        # --background-noise  \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_train  \
        # --pretrain \
        # out/hi_smart_v2/best-acc-0.958-cnn10_sgd_plateau_bs256_lr1.0e-03_wd1.0e-02-epoch12.pth \
        # --model                   cnn10  \
        # --optim                   sgd  \
        # --lr-scheduler            plateau  \
        # --dataload-workers-nums   8 \
        # --learning-rate           0.0001  \
        # --lr-scheduler-patience   10  \
        # --max-epochs              20  \
        # --batch-size              64  \
        # --input                   40  \
        # --save-path               out/hi_smart_v6    #微调，加video call前的最终版

        # --train-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/5m \
        # --valid-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/5m \
        # --background-noise  \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/train_class0  \
        # --pretrain \
        # out/hi_smart_v2/best-acc-0.958-cnn10_sgd_plateau_bs256_lr1.0e-03_wd1.0e-02-epoch12.pth \
        # --model                   cnn10  \
        # --optim                   sgd  \
        # --lr-scheduler            plateau  \
        # --dataload-workers-nums   8 \
        # --learning-rate           0.0001  \
        # --lr-scheduler-patience   10  \
        # --max-epochs              10  \
        # --batch-size              64  \
        # --input                   40  \
        # --save-path               out/hi_smart_v5


        # --train-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train/5m \
        # --valid-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val/5m \
        # --background-noise  \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/train_class0  \
        # --pretrain \
        # out/hi_smart_v2/best-acc-0.958-cnn10_sgd_plateau_bs256_lr1.0e-03_wd1.0e-02-epoch12.pth \
        # --model                   cnn10  \
        # --optim                   sgd  \
        # --lr-scheduler            plateau  \
        # --dataload-workers-nums   32 \
        # --learning-rate           0.0001  \
        # --lr-scheduler-patience   10  \
        # --max-epochs              10  \
        # --batch-size              64  \
        # --input                   40  \
        # --save-path               out/hi_smart_v4

        # --train-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/gen_commands_pad/inside_3s_8k \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_train \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/train_class0 \
        # /dataset/audio/audio_command/task_hi_smart/sppech_commands_neg  \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/1m \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/3m \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/5m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/1m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/3m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/5m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/speech_command_train \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/5m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_train/5m \
        # --valid-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/test \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/group_val/5m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_val/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_val/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_val/5m \


        # --train-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/train_class0 \
        # /dataset/audio/audio_command/task_hi_smart/sppech_commands_neg  \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/1m \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/3m \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/5m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/1m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/3m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/5m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/5m \
        # --valid-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/test \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/组员录音_重切_重命名_标准化/1m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/组员录音_重切_重命名_标准化/3m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/组员录音_重切_重命名_标准化/5m \
        # --background-noise  \
        # /dataset/audio/audio_command/collection_data/negative_sample_1226_linrui_cropped_normsound  \
        # --pretrain \
        # out/hi_smart_v1/best-acc-0.990-cnn10_sgd_plateau_bs256_lr1.0e-03_wd1.0e-02-epoch20.pth \
        # --model                   cnn10  \
        # --optim                   sgd  \
        # --lr-scheduler            plateau  \
        # --dataload-workers-nums   8 \
        # --learning-rate           0.01  \
        # --lr-scheduler-patience   10  \
        # --max-epochs              100  \
        # --batch-size              256  \
        # --input                   40  \
        # --save-path               out/hi_smart_v3







        # --train-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/gen_commands_pad/inside_3s_8k \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_train \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/train_class0 \
        # /dataset/audio/audio_command/task_hi_smart/sppech_commands_neg  \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/1m \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/3m \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/5m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/1m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/3m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/5m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/speech_command_train \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/5m \
        # --valid-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/test \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/组员录音_重切_重命名_标准化/1m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/组员录音_重切_重命名_标准化/3m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/组员录音_重切_重命名_标准化/5m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_val/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_val/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_val/5m \
        # --background-noise  \
        # /dataset/audio/audio_command/collection_data/negative_sample_1226_linrui_cropped_normsound  \
        # --pretrain \
        # out/hi_smart_v1/best-acc-0.990-cnn10_sgd_plateau_bs256_lr1.0e-03_wd1.0e-02-epoch20.pth \
        # --model                   cnn10  \
        # --optim                   sgd  \
        # --lr-scheduler            plateau  \
        # --dataload-workers-nums   8 \
        # --learning-rate           0.001  \
        # --lr-scheduler-patience   10  \
        # --max-epochs              100  \
        # --batch-size              256  \
        # --input                   40  \
        # --save-path               out/hi_smart_v2



        # --train-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/gen_commands_pad/inside_3s_8k \
        # /dataset/audio/audio_command/task_hi_smart/sentence_of_article_train \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/train_class0 \
        # /dataset/audio/audio_command/task_hi_smart/sppech_commands_neg  \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/1m \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/3m \
        # /dataset/audio/audio_command/task_hi_smart/commands_after_trans/5m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/speech_command_train \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/5m \
        # --valid-dataset  \
        # /dataset/audio/audio_command/task_hi_smart/yzx/speech_command_val \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/1m \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/5m \
        # /dataset/audio/audio_command/task_hi_smart/背景音_width2s/test \
        # /home/xueting.ma/语音数据的处理/自己采集的音频/组员录音_分割后标准化wav文件_未cover \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_val/1m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_val/3m \
        # /dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_val/5m \
        # --background-noise  \
        # /dataset/audio/audio_command/collection_data/negative_sample_1226_linrui_cropped_normsound  \
        # --pretrain \
        # /home/xueting.ma/语音数据的处理/pytorch-speech-commands/results/speech_command_hismart_turnon_shutdown/hismart_turnon_shutdown_class0intrainandval_trans_again_addtrain_去除训练集中的5m转录_再去除3m5m中的turnon转录_训练集中添加0类背景音/best-acc-0.923-cnn10_sgd_plateau_bs256_lr1.0e-02_wd1.0e-02-epoch0.pth \
        # --model                   cnn10  \
        # --optim                   sgd  \
        # --lr-scheduler            plateau  \
        # --dataload-workers-nums   8 \
        # --learning-rate           0.001  \
        # --lr-scheduler-patience   10  \
        # --max-epochs              100  \
        # --batch-size              256  \
        # --input                   40  \
        # --save-path               out/hi_smart_v1
