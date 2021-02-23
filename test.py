import torch
import numpy
from pathlib import Path
from torchvision import datasets, models, transforms
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
# from tools.plotcm import plot_confusion_matrix
# from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os
import pandas as pd
from cv2 import cv2
from my_dataloader import TestDataset
from tqdm import tqdm

device = torch.device("cuda")

PATH_test = r'D:\dataset\automatic-optical-defect-detection\org\test_images'    # 測試資料集
PATH_sample = r'D:\GitHub\Automatic-optical-defect-detection\results\upload_sample.csv' # 上傳範例檔案

model = torchvision.models.resnet101(pretrained=False, progress=True)
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, 6)
model.to(device)
model.load_state_dict(torch.load('./weights/rotate/epoch34-loss0.0000077-val_loss0.0179573-acc0.9970.pth'))    # 權重

test_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# 讀取測試順序
df = pd.read_csv(PATH_sample)
img_list = []
for index, row in df.iterrows():
    img = os.path.join(PATH_test, row['ID'])
    img_list.append(img)

# 載入測試資料
test_data = TestDataset(img_list, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# 測試
results = []                                     
model.eval()         
with torch.no_grad():                                 
    for data in tqdm(test_loader):
        data = data.to(device)
        pred = model(data)
        predict_y = torch.max(pred, dim=1)[1]
        results = np.hstack((results, np.array(predict_y.cpu().detach())))

df['Label'] = results
df.to_csv('./results/submission.csv', index=None)