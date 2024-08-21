import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.SpectralNet import SpectralNetModel


class MetaFewShotClustering(nn.Module):
    def __init__(self, img_dim, device, config):
        super(MetaFewShotClustering, self).__init__()
        self.config = config
        self.device = device
        self.img_dim = img_dim
        self.current_epoch = 0
        self.batch_size = self.config.get('batch_size')
        self.total_epoch = self.config.get('total_epoch')

        self.architecture = self.config["hiddens"]
        self.task_learning_rate = self.config.get("init_inner_loop_learning_rate")
        

        self.model = SpectralNetModel(
            self.architecture, self.img_dim
        )

        self.meta_scaler = nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU(inplace=True),
            nn.Linear(1, 1)
        ).to(device = self.device)
        
        self.optimizer = optim.Adam([
                        {'params': self.model.parameters()},
                        {'params': self.regularizer.parameters()},
                    ], lr=self.config.get("meta_learning_rate"), amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.total_epochs,
                                                              eta_min=self.config.get("min_learning_rate"))