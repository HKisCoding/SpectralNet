import pandas as pd
import numpy as np
import os
from PIL import Image

import torch 
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, image_dir, annotation, model):
        self.image_dir = image_dir
        self.annotation = pd.read_csv(annotation, index_col=0)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet= model
        self.resnet.to(self.device)
        self.resnet.eval()

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        img_name = self.annotation.iloc[index, 0]
        label_name = self.annotation.iloc[index, 1]
        label = self.annotation.iloc[index, 2]

        img_sub_dir = os.path.join(label_name, img_name)
        img_path = os.path.join (self.image_dir, img_sub_dir)
        img = Image.open(img_path).convert("RGB")

        image_tensor = self.transform(img).to(self.device)
        
        # Extract the ResNet features
        with torch.no_grad():
            features = self.resnet(image_tensor.unsqueeze(0)).cpu().numpy()[0]

        return features, label