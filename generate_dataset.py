# 依照圖片標籤分類至資料夾
import os
import pandas as pd
import shutil

csv_fold = r'D:\dataset\automatic-optical-defect-detection' # csv資料夾
img_fold = r'D:\dataset\automatic-optical-defect-detection\org'     # 原始資料集
generate_fold = r'D:\dataset\automatic-optical-defect-detection\generate_dataset'   # 資料集生成位置
list = ['train']


for item in list:
    path = os.path.join(csv_fold, item+'.csv')
    df = pd.read_csv(path)
    for index, row in df.iterrows():
        img_name = row['ID']
        label = str(row['Label'])
        print(img_name, label)
        img = os.path.join(img_fold, (item+'_images'), img_name)
        if not os.path.isdir(os.path.join(generate_fold, item, label)): #建立分類資料夾
            os.mkdir(os.path.join(generate_fold, item, label))
        shutil.copy(img, os.path.join(generate_fold, item, label, img_name))
   