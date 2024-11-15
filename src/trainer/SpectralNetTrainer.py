from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset

from sklearn.neighbors import kneighbors_graph

from utils.utils import *
from utils.loss import SpectralNetLoss
from src.models.SpectralNet import SpectralNetModel
from src.models.MetaScaleModel import MetaScaleModel


class SpectralNetTrainer():
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
        
        self.counter = 0
        self.siamese_net = siamese_net
        self.criterion = SpectralNetLoss()

        self.siamese_net = siamese_net
        self.spectral_net = SpectralNetModel(
            self.architecture, input_dim=self.X.shape[1]
        ).to(self.device)

        self.optimizer = optim.Adam(self.spectral_net.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        train_loader, ortho_loader, valid_loader = self._get_data_loader()

        print("Training SpectralNet:")
        t = trange(self.epochs, leave=True)
        total_train_loss = []
        total_val_loss = []
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
                self.spectral_net(X_orth, True)

                # Gradient step
                self.spectral_net.train()
                self.optimizer.zero_grad()

                Y = self.spectral_net(X_grad, False)
                if self.siamese_net is not None:
                    with torch.no_grad():
                        X_grad = self.siamese_net.single_forward(X_grad)

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
            total_train_loss.append(train_loss)
            total_val_loss.append(valid_loss)
            t.refresh()
        train_result = {"train_loss": total_train_loss, 
                        "val_loss": total_val_loss}
        
        # plot_loss(train_result)
        
        return self.spectral_net
    

    def metalearning_inner_train(self, X: torch.Tensor, y: torch.Tensor):
        self.counter = 0
        self.criterion = SpectralNetLoss()

        self.spectral_net = SpectralNetModel(
            self.architecture, input_dim=X.shape[1]
        ).to(self.device)

        self.meta_model = MetaScaleModel(input_dim = X.shape[1]).to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.spectral_net.parameters()},
            {'params': self.meta_model.parameters()}], 
            lr=self.lr, amsgrad=False)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        self.siamese_net = None
        print("Training SpectralNet:")
        t = trange(self.epochs, leave=True)
        total_train_loss = []
        total_val_loss = []
        for epoch in t:
            train_loss = 0.0
            # Flatten the input tensor
            X_support, y_support, X_target, y_target = self._meta_data_loader(X, y)

            scale = self.meta_model(X_support)
            scale = scale.view(-1).mean().item()

            self.X = X_target.view(X_target.size(0), -1)
            self.y = y_target
            train_loader, ortho_loader, valid_loader = self._get_data_loader()
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
                self.spectral_net(X_orth, True)

                # Gradient step
                self.spectral_net.train()
                self.optimizer.zero_grad()

                Y = self.spectral_net(X_grad, False)

                W = self.create_affingity_matrix_from_scale(X_grad, scale)

                loss = self.criterion(W, Y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation step
            valid_loss = self.meta_validate(valid_loader, scale)
            self.scheduler.step(valid_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.spectral_config["min_lr"]:
                break
            t.set_description(
                "Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(
                    train_loss, valid_loss, current_lr
                )
            )
            total_train_loss.append(train_loss)
            total_val_loss.append(valid_loss)
            t.refresh()
        train_result = {"train_loss": total_train_loss, 
                        "val_loss": total_val_loss}
        
        # plot_loss(train_result)
        
        return self.spectral_net
    

    def meta_validate(self, valid_loader: DataLoader, scale) -> float:
        valid_loss = 0.0
        self.spectral_net.eval()
        with torch.no_grad():
            for batch in valid_loader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)

                if self.is_sparse:
                    X = make_batch_for_sparse_grapsh(X)

                Y = self.spectral_net(X, False)
                with torch.no_grad():
                    if self.siamese_net is not None:
                        X = self.siamese_net.single_forward(X)
                
                W = self.create_affingity_matrix_from_scale(X, scale)

                loss = self.criterion(W, Y)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        return valid_loss
    

    def validate(self, valid_loader: DataLoader) -> float:
        valid_loss = 0.0
        self.spectral_net.eval()
        with torch.no_grad():
            for batch in valid_loader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)

                if self.is_sparse:
                    X = make_batch_for_sparse_grapsh(X)

                Y = self.spectral_net(X, False)
                with torch.no_grad():
                    if self.siamese_net is not None:
                        X = self.siamese_net.single_forward(X)

                W = self._get_affinity_matrix(X)

                loss = self.criterion(W, Y)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        return valid_loss
    

    def create_affingity_matrix_from_scale(self, X: torch.Tensor, scale: float) -> torch.Tensor:
        is_local = self.is_local_scale
        n_neighbors = self.n_nbg
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        # scale = compute_scale(Dis, k=scale, is_local=is_local)
        W = get_gaussian_kernel(
            Dx, scale, indices, device=self.device, is_local=is_local
        )
        return W
    

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
    

    def _get_data_loader(self) -> tuple:
        """
        This function returns the data loaders for training, validation and testing.

        Returns:
            tuple:  The data loaders
        """
        if self.y is None:
            self.y = torch.zeros(len(self.X))
        train_size = int(0.9 * len(self.X))
        valid_size = len(self.X) - train_size
        dataset = TensorDataset(self.X, self.y)
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        ortho_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_loader, ortho_loader, valid_loader
    

    def _meta_data_loader(self, X ,y) -> tuple:
        if y is None:
            y = torch.zeros(len(X))
        indices = torch.randperm(len(y))

        ratio = int(len(X) * 0.3)
    
        # Split indices
        support_indices = indices[:ratio]
        target_indices = indices[ratio:]
        
        # Create support and target sets
        X_support = X[support_indices]
        y_support = y[support_indices]
        X_target = X[target_indices]
        y_target = y[target_indices]
        
        return X_support, y_support, X_target, y_target
        
