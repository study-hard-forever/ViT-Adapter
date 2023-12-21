import cv2
import numpy as np
test_img = cv2.imread('data/VOCdevkit/VOC2007/test_mask/2_huahenC_3_huahenC_srcTray_1_srcIndex_2_DL_result_1_1_1_HuaShang.png')
print(np.unique(test_img))