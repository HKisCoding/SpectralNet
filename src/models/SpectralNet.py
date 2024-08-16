import torch 
import torch.nn as nn 
import numpy as np


class SpectralNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SpectralNetModel, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()
        self.input_dim = input_dim

        current_dim = self.input_dim
        for i, layer in enumerate(self.architecture):
            next_dim = layer
            if i == len(self.architecture) - 1:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                )
            else:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.LeakyReLU())
                )
                current_dim = next_dim


    def orthonormalize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Take the output of model and apply the orthonormalization using Cholesky decomposition 

        Args:
            input (torch.Tensor): output of gradient model.

        Returns:
            torch.Tensor: The orthonormalize weight after decomposition.
        """

        m = input.shape[0]
        _, L = torch.linalg.qr(input)
        w = np.sqrt(m) * torch.inverse(L)
        return w
    

    def forward(self, x: torch.Tensor, should_update_orth_weights: bool = True):
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): the input tensor
        
        Returns: 
            torch.Tensor: output tensor
        """

        for layer in self.layers:
            x = layer(x)

        Y_tilde = x
        if should_update_orth_weights:
            self.orthonorm_weights = self.orthonormalize(Y_tilde)

        Y = Y_tilde @ self.orthonorm_weights
        return Y

        

