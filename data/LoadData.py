import h5py
import numpy as np 

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms


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


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()