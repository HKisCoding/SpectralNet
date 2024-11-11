import torch
import numpy as np

from data.LoadData import load_mnist

from utils.metric import acc_score_metric, nmi_score_metric
from src.trainer.Trainer import SpectralNet


def main():
    x_train, y_train, x_test, y_test = load_mnist()

    X = torch.cat([x_train, x_test])

    if y_train is not None:
        y = torch.cat([y_train, y_test])
    else:
        y = None

    n_clusters  = len(torch.unique(y))

    spectralnet = SpectralNet(
        n_clusters=n_clusters,
        should_use_ae=True,
        should_use_siamese=True,
        ae_hiddens = [512, 512, 2048, n_clusters],
        siamese_epochs = 100,
        siamese_hiddens = [1024, 1024, 512, n_clusters],
        spectral_epochs= 500,
        # spectral_scale_k= 20,
        spectral_is_local_scale = True,
        spectral_batch_size = 2000,
        spectral_hiddens = [1024, 1024, 512, n_clusters]
        # ae_epochs= 5,
        # siamese_epochs= 5,
        # spectral_epochs=5
    )
    spectralnet.fit(X, y)
    cluster_assignments = spectralnet.predict(X)
    embeddings = spectralnet.embeddings_

    if y is not None:
        y = y.detach().cpu().numpy()
        acc_score = acc_score_metric(cluster_assignments, y, n_clusters=n_clusters)
        nmi_score = nmi_score_metric(cluster_assignments, y)
        print(f"ACC: {np.round(acc_score, 3)}")
        print(f"NMI: {np.round(nmi_score, 3)}")

    return embeddings, cluster_assignments


if __name__ == "__main__":
    embeddings, assignments = main()