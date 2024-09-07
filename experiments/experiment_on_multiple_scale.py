import torch
import numpy as np
import pandas as pd

from data.LoadData import get_feature_labels, get_prokaryotic, load_mnist

from utils.metric import acc_score_metric, nmi_score_metric, purity_score_metrics

from src.trainer.Trainer import SpectralNet


def run_cluster(X, y, k, is_global = False):
    n_clusters  = len(torch.unique(y))
    spectralnet = SpectralNet(
        n_clusters=n_clusters,
        should_use_ae=False,
        should_use_siamese=False,
        ae_hiddens = [512, 512, 2048, n_clusters],
        siamese_epochs = 100,
        siamese_hiddens = [1024, 1024, 512, n_clusters],
        spectral_epochs= 200,
        spectral_scale_k= 20,
        spectral_is_local_scale = False,
        spectral_batch_size = 1024,
        spectral_hiddens = [1024, 1024, 512, n_clusters]
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

    return np.round(acc_score, 3), np.round(nmi_score, 3), np.round(purity_score, 3)


if __name__ == "__main__":
    gauss_scale = [5, 10, 20, 30, 50]
    feature_path = "dataset/MRSC_Feature.pt"
    labels_path = "dataset/MRSC_Label.pt"
    X, y = get_feature_labels(feature_path, labels_path)
    res = []
    for scale in gauss_scale:
        acc, nmi, purity = run_cluster(X, y, scale)
        res.append({
            'gauss_scale': scale,
            'acc': acc,
            'nmi': nmi,
            'purity': purity
        })
    acc, nmi, purity = run_cluster(X, y, scale, True)
    res.append({
        'gauss_scale': 'Full neightbor',
        'acc': acc,
        'nmi': nmi,
        'purity': purity
    })
    df = pd.DataFrame(res)
    df.to_csv('spectralnet/output/MRSC_gauss_scale_comparision.csv')
