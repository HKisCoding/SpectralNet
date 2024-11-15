import torch
import numpy as np

from data.LoadData import *

from utils.metric import acc_score_metric, nmi_score_metric, purity_score_metrics
# from spectralnet import SpectralNet
from src.trainer.Trainer import SpectralNet
from experiments.experiment_config import ExperimentConfig


def main():
    X, y = get_colon_cancer("dataset\colon_cancer\colon_cancer.csv")
    n_clusters  = len(torch.unique(y))
    spectral_config = ExperimentConfig(n_clusters, k =20)
    spectralnet = SpectralNet(
        **spectral_config
    )
    spectralnet.fit(X, y)
    cluster_assignments = spectralnet.predict(X)
    embeddings = spectralnet.embeddings_

    if y is not None:
        y = y.detach().cpu().numpy()
        acc_score = acc_score_metric(cluster_assignments, y, n_clusters=n_clusters)
        nmi_score = nmi_score_metric(cluster_assignments, y)
        purity_score = purity_score_metrics(cluster_assignments, y)
        print(f"ACC: {np.round(acc_score, 3)}")
        print(f"NMI: {np.round(nmi_score, 3)}")
        print (f"PURITY: {np.round(purity_score, 3)}")

    return embeddings, cluster_assignments


if __name__ == "__main__":
    embeddings, assignments = main()
