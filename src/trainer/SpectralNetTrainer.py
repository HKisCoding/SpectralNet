from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from sklearn.neighbors import kneighbors_graph

from utils.utils import *
from models.SpectralNet import SpectralNet


class Trainer():
    def __init__(self, config, device, is_sparse):
        self.device = device
        self.is_sparse = is_sparse
        self.spectral_config = config
        self.lr = self.spectral_config["lr"]
        self.n_nbg = self.spectral_config["n_nbg"]
        self.min_lr = self.spectral_config["min_lr"]
        self.epochs = self.spectral_config["epochs"]
        self.scale_k = self.spectral_config["scale_k"]
        self.lr_decay = self.spectral_config["lr_decay"]
        self.patience = self.spectral_config["patience"]
        self.architecture = self.spectral_config["hiddens"]
        self.batch_size = self.spectral_config["batch_size"]
        self.is_local_scale = self.spectral_config["is_local_scale"]


    def train(
        self, X: torch.Tensor, y: torch.Tensor, siamese_net: nn.Module = None
    ):
        # Flatten the input tensor
        self.X = X.view(X.size(0), -1)
        self.y = y

        self.siamese_net = siamese_net
        self.spectral_net = SpectralNet(
            self.architecture, input_dim=self.X.shape[1]
        ).to(self.device)

        self.optimizer = optim.Adam(self.spectral_net.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        train_loader, ortho_loader, valid_loader = self._get_data_loader()

        print("Training SpectralNet:")
        t = trange(self.epochs, leave=True)
        for epoch in t:
            train_loss = 0.0
            for (X_grad, _), (X_orth, _) in zip(train_loader, ortho_loader):
                X_grad = X_grad.to(device=self.device)
                X_grad = X_grad.view(X_grad.size(0), -1)
                X_orth = X_orth.to(device=self.device)
                X_orth = X_orth.view(X_orth.size(0), -1)

                if self.is_sparse:
                    X_grad = make_batch_for_sparse_grapsh(X_grad)
                    X_orth = make_batch_for_sparse_grapsh(X_orth)

                # Orthogonalization step
                self.spectral_net.eval()
                self.spectral_net(X_orth, should_update_orth_weights=True)

                # Gradient step
                self.spectral_net.train()
                self.optimizer.zero_grad()

                Y = self.spectral_net(X_grad, should_update_orth_weights=False)
                if self.siamese_net is not None:
                    with torch.no_grad():
                        X_grad = self.siamese_net.forward_once(X_grad)

                W = self._get_affinity_matrix(X_grad)

                loss = self.criterion(W, Y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation step
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(valid_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.spectral_config["min_lr"]:
                break
            t.set_description(
                "Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(
                    train_loss, valid_loss, current_lr
                )
            )
            t.refresh()

        return self.spectral_net
    

    def _get_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        This function computes the affinity matrix W using the Gaussian kernel.

        Args:
            X (torch.Tensor):   The input data

        Returns:
            torch.Tensor: The affinity matrix W
        """

        is_local = self.is_local_scale
        n_neighbors = self.n_nbg
        scale_k = self.scale_k
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        scale = compute_scale(Dis, k=scale_k, is_local=is_local)
        W = get_gaussian_kernel(
            Dx, scale, indices, device=self.device, is_local=is_local
        )
        return W
