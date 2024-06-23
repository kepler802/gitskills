import os
import glob
import random




def ln_err():

    txt_path = "hi_smarts/err/Apr30_22-43-33.txt"
    dst_dir = "tmp_test/{}_err".format(txt_path.split("/")[-3]+ "_" + txt_path.split("/")[-1].split('.')[0])

    os.makedirs(dst_dir, exist_ok=True)

    with open(txt_path, "r") as f:
        lines = f.read().splitlines()

    for line in lines:
        path = line.split(" ")[0]

        dst_path = os.path.join(dst_dir, "#".join(path.split("/")[-4:]))

        os.symlink(path, dst_path)


# ln_err()






def ln_train_data(src_dir_list, dst_dir):

    num_per_class = 50


    for src_dir in src_dir_list:

        class_list = os.listdir(src_dir)

        for cla in class_list:

            dst_save_dir = os.path.join(dst_dir, src_dir.split("/")[-1], cla)
            os.makedirs(dst_save_dir, exist_ok=True)

            class_dir = os.path.join(src_dir, cla)

            file_list = glob.glob(class_dir + "/**/*.*", recursive=True)
            random.shuffle(file_list)

            print(class_dir, ":  ", len(file_list))

            for f in file_list[:num_per_class]:

                dst_name = "#".join(f.split("/")[-2:])

                dst_path = os.path.join(dst_save_dir, dst_name)

                os.symlink(f, dst_path)



src_dir_list = [
# "/dataset/audio/audio_command/task_hi_smart/gen_commands_pad/inside_3s_8k",
# "/dataset/audio/audio_command/task_hi_smart/sentence_of_article_train",
# "/dataset/audio/audio_command/task_hi_smart/sppech_commands_neg",
# "/dataset/audio/audio_command/task_hi_smart/commands_after_trans/1m",
# "/dataset/audio/audio_command/task_hi_smart/commands_after_trans/3m",
# "/dataset/audio/audio_command/task_hi_smart/commands_after_trans/5m",
# "/dataset/audio/audio_command/task_hi_smart/背景音_width2s/train_class0",


# "/dataset/audio/audio_command/task_hi_smart/背景音_width2s/train_class0", 
# "/dataset/audio/audio_command/task_hi_smart/sppech_commands_neg" ,
# "/dataset/audio/audio_command/task_hi_smart/commands_after_trans/1m" ,
# "/dataset/audio/audio_command/task_hi_smart/commands_after_trans/3m",
# "/dataset/audio/audio_command/task_hi_smart/commands_after_trans/5m" ,
"/home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/1m" ,
"/home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/3m" ,
"/home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/5m" ,
# "/dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/1m" ,
# "/dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/3m" ,
# "/dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/5m" ,


]

dst_dir = "/home/zhenxing.ye/work/pytorch-speech-commands/ln_data/hi_smart_train"

# src_dir_list = [

#     "/dataset/audio/audio_command/task_hi_smart/speech_command_val_8k",
#     "/dataset/audio/audio_command/task_hi_smart/sentence_of_article_8k/1m",
#     "/dataset/audio/audio_command/task_hi_smart/sentence_of_article_8k/3m",
#     "/dataset/audio/audio_command/task_hi_smart/sentence_of_article_8k/5m",

#     "/home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/1m" ,
#     "/home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/3m",
#     "/home/xueting.ma/语音数据的处理/自己采集的音频/代码生成的val指令_转录后分割/5m" ,
#     "/dataset/audio/audio_command/task_hi_smart/背景音_width2s/test",
#     "/home/xueting.ma/语音数据的处理/自己采集的音频/组员录音_分割后标准化wav文件_未cover"    ,

# ]

# dst_dir ="/home/zhenxing.ye/work/pytorch-speech-commands/ln_data/hi_smart_val"





# ln_train_data(src_dir_list, dst_dir)



def ln_data(src_dir, train_dir, val_dir, train_ratio=None, select_id=None):

    dir_list = os.listdir(src_dir)

    for dir in dir_list:

        train_sub = os.path.join(train_dir, dir)
        os.makedirs(train_sub, exist_ok=True)

        val_sub = os.path.join(val_dir, dir)
        os.makedirs(val_sub, exist_ok=True)

        dir_path = os.path.join(src_dir, dir)

        file_list = os.listdir(dir_path)

        random.shuffle(file_list)

        if train_ratio != None:
            for i in range(len(file_list)):
                src_path = os.path.join(dir_path, file_list[i])

                if i < int(len(file_list) * train_ratio):
                    dst_path = os.path.join(train_sub, file_list[i])
                else:
                    dst_path = os.path.join(val_sub, file_list[i])

                os.symlink(src_path, dst_path)

        else:
            assert select_id != None
            for i in range(len(file_list)):
                src_path = os.path.join(dir_path, file_list[i])

                id = int(src_path.split("/")[-1].split("_")[-2])

                # if i % 2 == 0:
                if id in select_id:
                    dst_path = os.path.join(train_sub, file_list[i])
                else:
                    dst_path = os.path.join(val_sub, file_list[i])

                os.symlink(src_path, dst_path)


# src_dir = "/dataset/audio/audio_command/task_hi_smart/speech_command_val_8k"
# train_dir = "/dataset/audio/audio_command/task_hi_smart/yzx/speech_command_train"
# val_dir = "/dataset/audio/audio_command/task_hi_smart/yzx/speech_command_val"

# src_dir = "/dataset/audio/audio_command/task_hi_smart/sentence_of_article_8k/5m"
# train_dir = "/dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_train/5m"
# val_dir = "/dataset/audio/audio_command/task_hi_smart/yzx/sentence_of_article_val/5m"


for i in [1,3,5]:

    # src_dir = "/dataset/audio/audio_command/task_hi_smart/group_collection/{}m".format(i)
    # train_dir = "/dataset/audio/audio_command/task_hi_smart/yzx/group_train_0.3/{}m".format(i)
    # val_dir = "/dataset/audio/audio_command/task_hi_smart/yzx/group_val_0.7/{}m".format(i)
    # ln_data(src_dir, train_dir, val_dir, select_id=[2,5,8])


    # src_dir = "/dataset/audio/audio_command/task_hi_smart/article_val_zhuanlu_bg/{}m".format(i)
    # train_dir = "/dataset/audio/audio_command/task_hi_smart/yzx/article_zhuanlu_train/{}m".format(i)
    # val_dir = "/dataset/audio/audio_command/task_hi_smart/yzx/article_zhuanlu_val/{}m".format(i)   
    # ln_data(src_dir, train_dir, val_dir, train_ratio=0.6)


    src_dir = "/dataset/audio/audio_command/task_hi_smart/gen_RTVC_zhuanlu/{}m".format(i)
    train_dir = "/dataset/audio/audio_command/task_hi_smart/yzx/RTVC_zhuanlu_train/{}m".format(i) 
    val_dir = "/dataset/audio/audio_command/task_hi_smart/yzx/RTVC_zhuanlu_val/{}m".format(i)    
    ln_data(src_dir, train_dir, val_dir, train_ratio=0.6)

















