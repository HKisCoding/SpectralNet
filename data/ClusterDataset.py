import pandas as pd
import os
from PIL import Image

import torch 
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, image_dir, annotation):
        self.image_dir = image_dir
        self.annotation = pd.read_csv(annotation)
        self.transform = transforms.Compose([
            transforms.PILToTensor()
        ])

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        img_name = self.annotation.iloc[index, 0]
        label = self.annotation.iloc[index, 1]
        img_sub_dir = os.path.join(img_name, label)
        img_path = os.path.join (self.image_dir, img_sub_dir)
        img = Image.open(img_path)

        return self.transform(img), label

        
    