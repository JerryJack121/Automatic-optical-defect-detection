from torch.utils.data.dataset import Dataset
from PIL import Image

class TestDataset(Dataset):
    def __init__(self, img_list, transform):
        super(TestDataset, self).__init__()
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        trans_img = self.transform(img)
        return trans_img
        
    