import torch
from pathlib import Path
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
# from model import CNN_Model
# from ResNet18 import resnet18
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import os

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda")

train_path = r'D:\dataset\automatic-optical-defect-detection\generate_dataset\augment\train'
val_path = r'D:\dataset\automatic-optical-defect-detection\generate_dataset\val'

# number of subprocesses to use for data loading
num_workers = 0
# 設計超參數
learning_rate = 0.0001
weight_decay = 0
epochs = 2
batch_size = 16

# convert data to a normalized torch.FloatTensor
train_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

val_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(Path(train_path), transform=train_transforms)
val_data = datasets.ImageFolder(Path(val_path), transform=val_transforms)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=batch_size,  num_workers=num_workers, shuffle=True)

# model = torch.load('resnet18.pt')
model = torchvision.models.resnet101(pretrained=True, progress=True)
# 提取參數fc的輸入參數
fc_features = model.fc.in_features
# 將最後輸出類別改為20
model.fc = nn.Linear(fc_features, 6)

# 驗證資料數
val_num = len(val_loader.dataset)

# 遷移學習 -> frezee
# for name, parameter in model.named_parameters():
#     # print(name)
#     if name == 'layer4.0.conv1.weight':
#         break
#     # if name == 'fc.weight':
#     #     break
#     parameter.requires_grad = False

# 載入預訓練權重
model.load_state_dict(torch.load('./weights/0202/epoch9-loss0.0058640-val_loss0.0170710-acc0.9970.pth'))
model.to(device)

# 定義損失函數
# criterion = nn.MSELoss(reduction='mean')
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
# 定義優化器
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)

# 學習率下降
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
train_losses, val_losses = [], []

loss_list = []
val_loss_list = []
val_acc_list = []
# train
for epoch in range(1, epochs+1):
    total_loss = 0.0
    total_val_loss = 0.0
    print('\nrunning epoch: {}'.format(epoch))
    # 訓練模式
    model.train()
    with tqdm(train_loader) as pbar:
        for inputs, target in train_loader:
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            running_loss = criterion(outputs, target)
            total_loss += running_loss.item()*inputs.size(0)
            running_loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_description('train')
            pbar.set_postfix(
                **{
                    'epoch': str('{}/{}'.format(epoch, epochs)),
                    'loss': running_loss.item(),
                    'lr': optimizer.state_dict()['param_groups'][0]['lr']
                })
    scheduler.step()

    # 驗證模式
    model.eval()
    total_val_acc = 0
    with torch.no_grad():
        with tqdm(val_loader) as pbar:
            for inputs, target in val_loader:
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs)
                running_val_loss = criterion(outputs, target)
                total_val_loss += running_val_loss.item()*inputs.size(0)
                predict_y = torch.max(outputs, dim=1)[1]
                total_val_acc += (predict_y == target).sum().item()
                pbar.update(1)
                pbar.set_description('valid')
                pbar.set_postfix(
                **{
                    'epoch': str('{}/{}'.format(epoch, epochs)),
                    'val_loss': running_val_loss.item(),
                })
    val_acc = total_val_acc / val_num
    loss = total_loss/len(train_loader.dataset)
    val_loss = total_val_loss/len(val_loader.dataset)
    loss_list.append(loss)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    print('Loss: {:.6f} \tval_Loss: {:.6f} \tacc: {:.6f} '.format(loss, val_loss, val_acc))
    torch.save(model.state_dict(), './logs/epoch%d-loss%.7f-val_loss%.7f-acc%.4f.pth' %(epoch, loss, val_loss, val_acc))

# 繪製圖
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(loss_list, label='train_losses')
plt.plot(val_loss_list, label='val_losses')
plt.legend(loc='best')
plt.savefig('./images/losses__StepLR_5.jpg')
plt.show()
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.plot(val_acc_list, label='val_acc')
plt.legend(loc='best')
plt.savefig('./images/acc.jpg')
plt.show()