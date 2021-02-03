from torch.utils.data.dataset import Dataset
from PIL import Image

class TestDataset(Dataset):
    def __init__(self, img_list, transform, device):
        self.data = []
        for img_name in img_list:
            img = Image.open(img_name).convert('RGB')
            img = transform(img)
            self.data.append(img)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
        
    