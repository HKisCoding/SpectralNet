import torch
import torch.nn


def loss_function(W: torch.Tensor, Y: torch.Tensor, is_normalized: bool):
    m = Y.size(0)
    if is_normalized:
        D = torch.sum(W, dim=1)
        Y = Y / torch.sqrt(D)[:, None]

    Dy = torch.cdist(Y, Y)
    loss = torch.sum(W * Dy.pow(2)) / (2 * m)

    return loss