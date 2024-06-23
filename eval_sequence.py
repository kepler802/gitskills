import os



with open(r"\\10.80.39.3\zhenxing.ye\nfsuser\infer\sound\output\xiaoai_v23_sequence.txt", "r") as f:

    lines = f.read().splitlines()

for line in lines:

    res = [int(i) for i in line.split(" ")[1: ]]

    if res[0] != 0 and res[0] != 1:
        print(line)
        continue

    cur = 0
    cur_list = [0]

    for i in range(len(res)):
        if res[i] != cur and res[i] != 0:

            if res[i] == len(cur_list):

                cur = res[i]
                cur_list.append(res[i])

            else:
                print(line)


                
aaa=0




