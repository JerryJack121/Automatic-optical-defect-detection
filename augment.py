import os
from cv2 import cv2
from PIL import Image
import numpy as np
# import argparse
import random

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

def avg_blur(img, max_filiter_size = 3) :
	img = img.astype(np.uint8)
	if max_filiter_size >= 3 :
		filter_size = random.randint(3, max_filiter_size)
		if filter_size % 2 == 0 :
			filter_size += 1
		out = cv2.blur(img, (filter_size, filter_size))
	return out

def gaussain_blur(img, max_filiter_size = 3, sigma = 0) :
    img = img.astype(np.uint8)
    if max_filiter_size >= 3 :
        filter_size = random.randint(3, max_filiter_size)
        if filter_size % 2 == 0 :
            filter_size += 1
    # print ('size = %d'% filter_size)
    out = cv2.GaussianBlur(img, (filter_size, filter_size), sigma)
    return out

def flip(img) :
	img = img.astype(np.uint8)
	flip_factor = random.randint(-1, 1) # -1:水平垂直翻轉，0:垂直翻轉，1:水平翻轉
	out = cv2.flip(img, flip_factor)
	return out, flip_factor

# input_path
input_path = r'D:\dataset\automatic-optical-defect-detection\generate_dataset\train'  # 輸入資料夾
# output_path
out_path = r'D:\dataset\automatic-optical-defect-detection\generate_dataset\train_aug'
# 擴增後每一類別的數量
total_num = 1000
aug_list = ['flip', 'guss']
fold = '0'
fold_path = os.path.join(input_path, fold)
img_list = os.listdir(fold_path)
if len(img_list) < total_num:    # 判斷資料是否需要擴增
    idxs = np.random.randint(0, len(img_list), size= total_num - len(img_list)) # 隨機挑選被擴增
    for idx in  idxs:
        img_name = img_list[idx]
        img = cv2.imread(os.path.join(fold_path, img_name))
        aug = random.choice(aug_list)
        if  aug == 'flip':   # 翻轉
            img_flip, flip_factor = flip(img)
            cv2.imshow('org', img)
            cv2.imshow('flip', img_flip)
            cv2.waitKey(0)
        elif aug == 'guss': # 高斯濾波
            img_guss = gaussain_blur(img)
            cv2.imshow('org', img)
            cv2.imshow('blur', img_guss)
            cv2.waitKey(0)
        elif aug == 'avg': # 均值濾波
            img_blur = avg_blur(img)
            cv2.imshow('org', img)
            cv2.imshow('blur', img_blur)
            cv2.waitKey(0)
        elif aug == 'rotate': # 旋轉
            img_blur = avg_blur(img)
            img_rotate = rotate(img , degree)
            cv2.imshow(img_name, img_rotate)
            cv2.waitKey(0)

       

        

        
        