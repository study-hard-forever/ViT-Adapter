import cv2
import os
import shutil
from PIL import Image

img_path = r"segmentation/data/VOCdevkit/VOC2007/SegmentationClass_old"
'''
...
图像通道数：RGB
图像通道数：L
图像通道数：L
图像通道数：L
图像通道数：RGB
图像通道数：L
图像通道数：L
图像通道数：L
图像通道数：L
总数量： 5105
不符合标准规范的： 409
'''
# img_path_target = r"segmentation/data/VOCdevkit/VOC2007/SegmentationClass"
# os.makedirs(img_path_target, exist_ok=True)
"""
判断是否符合单通道标准，结果发现原来的mask图像中存在3通道图像，因此后续在加载mask时会导致维度对应不上
"""
count = 0
count_error = 0  # 不符合单通道标准规范的
for i in os.listdir(img_path):
    count += 1
    '''此段脚本无用，改为PIL读取图像了
    # 读取图像
    # cv2默认读取的是彩色图像，因此即使图像是L灰度图模式，也会变成3通道BGR格式，因此此处改为PIL Image进行图像读取
    image = cv2.imread(os.path.join(img_path, i))

    # 获取图像形状
    height, width, channels = image.shape
    """"""
    if channels != 1:
        count_error += 1
        # 检查三个通道是否相等  # 经判断都是一致的
        channel_0 = image[:, :, 0]
        channel_1 = image[:, :, 1]
        channel_2 = image[:, :, 2]

        # 判断是否所有通道值相等
        are_channels_equal = (channel_0 == channel_1).all() and (
            channel_1 == channel_2
        ).all()

        if are_channels_equal:
            pass
            print("图像的三个通道的值相等。")
        else:
            print("图像的三个通道的值不相等。")
        print(f"图像高度：{height}, 图像宽度：{width}, 通道数：{channels}, 图像名称：{i}")
    '''
    
    # 打开图像
    image = Image.open(os.path.join(img_path, i))

    # 获取图像模式（通道数）
    mode = image.mode
    if mode != "L":
        count_error += 1
        
    # 打印通道数
    print(f"图像通道数：{mode}")
    
    """
    # 判断通道数并将不规范的数据转换为规范的移动到新的文件夹中
    if image.shape[2] == 1:  # 单通道
        dest_path = os.path.join(img_path_target, i)
        shutil.copy(os.path.join(img_path, i), dest_path)
    else:  # 多通道
        # 转换为单通道（灰度图）
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = image[:, :, 0]

        dest_path = os.path.join(img_path_target, i)
        cv2.imwrite(dest_path, gray_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 用于强制保存灰度图格式
    """
print(f'总数量： {count}')
print(f'不符合标准规范的： {count_error}')
