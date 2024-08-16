import torch
import numpy as np

from data.LoadData import load_mnist

from utils.metric import acc_score, nmi_score
from spectralnet import SpectralNet


def main():
    x_train, y_train, x_test, y_test = load_mnist()

    X = torch.cat([x_train, x_test])

    if y_train is not None:
        y = torch.cat([y_train, y_test])
    else:
        y = None

    spectralnet = SpectralNet(
        n_clusters=10,
        should_use_ae=True,
        should_use_siamese=True,
    )
    spectralnet.fit(X, y)
    cluster_assignments = spectralnet.predict(X)
    embeddings = spectralnet.embeddings_

    if y is not None:
        y = y.detach().cpu().numpy()
        acc_score = acc_score(cluster_assignments, y, n_clusters=10)
        nmi_score = nmi_score(cluster_assignments, y)
        print(f"ACC: {np.round(acc_score, 3)}")
        print(f"NMI: {np.round(nmi_score, 3)}")

    return embeddings, cluster_assignments


if __name__ == "__main__":
    embeddings, assignments = main()