import h5py
from scipy.io import loadmat 
import numpy as np 
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms

from data.ClusterDataset import ImageDataset


def load_mnist():
    transform_data = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        root = "dataset", train = True, download= True, transform= transform_data
    )

    test_set = datasets.MNIST(
        root = "dataset", train = False, download= True, transform= transform_data
    )

    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)


    return x_train, y_train, x_test, y_test


def load_Caltech():
    img_dir = "dataset\\Caltech_101"
    annotation = "dataset\\Caltech_101.csv"

    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    dataset = ImageDataset(img_dir, annotation, resnet)

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size = 0.3)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    
    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*val_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)


    return x_train, y_train, x_test, y_test


def get_feature_labels(feature_path, labels_path):
    X = torch.load(feature_path)
    y = torch.load(labels_path)

    return X, y


def get_prokaryotic(path):
    data = loadmat(r'dataset/prokaryotic.mat')
    label = data['truth'][:,0]
    feature = data['gene_repert']

    return torch.tensor(feature).float(), torch.Tensor(label)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_Caltech()