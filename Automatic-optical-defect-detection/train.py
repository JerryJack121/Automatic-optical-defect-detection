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

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

train_path = r'D:\dataset\automatic-optical-defect-detection\generate_dataset\train'
val_path = r'D:\dataset\automatic-optical-defect-detection\generate_dataset\val'

# number of subprocesses to use for data loading
num_workers = 0
# 設計超參數
learning_rate = 0.0001
weight_decay = 0
EPOCH = 100
batch_size = 30
val_batch_size = 30

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
valid_loader = torch.utils.data.DataLoader(
    val_data, batch_size=val_batch_size,  num_workers=num_workers, shuffle=True)