import os
from cv2 import cv2
from PIL import Image
import numpy as np
import argparse

def rotate(image, angle, center=None, scale=1.0):
    # 圖片寬、高
    (h, w) = image.shape[:2]
 
    # 預設使用圖片中心作為旋轉中心
    if center is None:
        center = (w / 2, h / 2)
 
    # 旋轉
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    return rotated

# input_path
img_path = r'D:\dataset\automatic-optical-defect-detection\generate_dataset\train'  # 輸入資料夾
# output_path
out_path = r'D:\dataset\automatic-optical-defect-detection\generate_dataset\rotate_train'

for fold in os.listdir(img_path):
    fold_path = os.path.join(img_path, fold)
    for img_name in os.listdir(fold_path):
        img_path = os.path.join(fold_path, img_name)
        img = cv2.imread(img_path)
        #圖片旋轉
        for degree in range(-60, 75, 15):
            print(degree)
            img_rotate = rotate(img , degree)
            cv2.imshow(img_name, img_rotate)
            cv2.waitKey(0)