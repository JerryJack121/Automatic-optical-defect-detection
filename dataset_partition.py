#從訓練資料夾中挑選驗證資料
import os
from random import sample
import shutil

# 訓練集路徑
train_fold = r'D:\dataset\automatic-optical-defect-detection\generate_dataset\train'   
# 生成驗證集路徑
val_fold = r'D:\dataset\automatic-optical-defect-detection\generate_dataset\val'   
# 訓練集佔總資料集的比例
training_rate = 0.6


#創建驗證資料夾
if not os.path.isdir(val_fold):
    os.mkdir(val_fold)

img_num = 0
fold_list = os.listdir(train_fold)

# 計算原始資料集數量
for fold in fold_list:
    fold_img_list = os.listdir(os.path.join(train_fold, fold))
    img_num += len(fold_img_list)

val_num = int(img_num* (1 - training_rate))

for fold in fold_list:
    img_list = []
    fold_img_list = os.listdir(os.path.join(train_fold, fold))
    for img in fold_img_list:   #讀取類別內所有圖片
        img_path = os.path.join(fold, img)
        img_list.append(img_path)
    if not os.path.isdir(os.path.join(val_fold, fold)): #建立驗證資料夾
        os.mkdir(os.path.join(val_fold, fold))
    print(int(val_num*(len(img_list)/img_num)))
    val_img_list = sample(img_list, int(val_num*(len(img_list)/img_num)))
    for val_img in val_img_list:
        print(val_img)
        shutil.move(os.path.join(train_fold, val_img), os.path.join(val_fold, val_img))
    
print('總數: ', img_num)

print('驗證數量: ', val_num)