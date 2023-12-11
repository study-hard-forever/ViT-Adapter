# -*- coding: utf-8 -*-
# @Author      : MJG
# FileName     : 
# @Time        : 
# @Description : 多线程统计像素值
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

mask_path = r"segmentation/data/VOCdevkit/VOC2007/SegmentationClass"

# 统计mask像素值
mask_category_set = set()
def unique_pixels(mask_file):
  mask = cv2.imdecode(np.fromfile(mask_file, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
  # 查看mask中都有哪些像素
  unique_pixels = np.unique(mask)
  mask_category_set.update(unique_pixels)
  print(mask_file, unique_pixels)
     
if __name__ == "__main__": 
  with ThreadPoolExecutor() as executor:
    # 遍历文件夹中的所有文件，提交每个文件的处理任务到线程池
    futures = {executor.submit(unique_pixels, os.path.join(mask_path, mask_file)): mask_file for mask_file in os.listdir(mask_path)}

    # 等待所有任务完成
    for future in tqdm(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Error processing file: {futures[future]} - {e}")
  print(mask_category_set)
    
