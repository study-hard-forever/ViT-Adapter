# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random
import os

# calculate means and std  注意换行\n符号
path = "segmentation/data/VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt"  # 先生成记录图片数据路径的txt文件
means = [0, 0, 0]
stdevs = [0, 0, 0]

index = 1
num_imgs = 0
print("计算中，这可能会花费一两分钟...")
with open(path, "r") as f:
    lines = f.readlines()
    # random.shuffle(lines)
    times = 1.5  # 计算量倍数（此处不是单纯的按照所有图像逐一进行计算，而是对所有图像进行随机采样，此处采样率为所有图像的1.5倍
    for index in range((int)(len(lines) * times)):
        x = random.randint(0, len(lines) - 1)
        line = lines[x]
        # print(line)
        # print('{}/{}'.format(index, len(lines)))
        index += 1
        a = os.path.join("segmentation/data/VOCdevkit/VOC2007/JPEGImages/" + line)
        # print(a[:-1])
        num_imgs += 1
        img = cv2.imread(a[:-1] + ".jpg")
        img = np.asarray(img)
        img = img.astype(np.float32) / 255.0
        for i in range(3):
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()

means.reverse()  #  将opencv的BGR格式调整为RGB格式
stdevs.reverse()

means = np.asarray(means) / num_imgs
stdevs = np.asarray(stdevs) / num_imgs

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print("transforms.Normalize({},{})".format(means, stdevs))



'''
segmentation/data/VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt
normMean = [0.63922347 0.71380336 0.62241851]
normStd = [0.18011222 0.18052031 0.17824703]
transforms.Normalize([0.63922347 0.71380336 0.62241851],[0.18011222 0.18052031 0.17824703])

segmentation/data/VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt
normMean = [0.63620229 0.71192961 0.6202118 ]
normStd = [0.17900929 0.18039188 0.17705131]
transforms.Normalize([0.63620229 0.71192961 0.6202118 ],[0.17900929 0.18039188 0.17705131])

mmsegment默认以255单位进行划分，因此需要乘上255
参考链接：https://blog.csdn.net/zylooooooooong/article/details/122805833
默认ImageNet的：均值：image_mean=[0.485,0.456,0.406]，标准差：image_std=[0.229,0.224,0.225]
对应mmsegment的：# mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

因此此处训练的修改如下：

normMean = [0.63922347,0.71380336,0.62241851]
normStd = [0.18011222,0.18052031,0.17824703]
print([i * 255 for i in normMean])
print([i * 255 for i in normStd])
# 结果为：
[163.00198485, 182.0198568, 158.71672005000002]
[45.9286161, 46.03267905, 45.45299265]
'''
