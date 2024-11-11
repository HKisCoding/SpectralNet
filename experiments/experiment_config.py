def autoencoder_with_siamese(n_clusters):
    return  {
        "n_clusters": n_clusters,
        "should_use_ae" : True,
        "should_use_siamese" : True,
        "ae_hiddens" : [512, 512, 2048, n_clusters],
        "siamese_epochs" : 100,
        "siamese_hiddens" : [1024, 1024, 512, n_clusters],
        "spectral_epochs" : 500,
        # spectral_scale_k= 20,
        "spectral_is_local_scale" : True,
        "spectral_batch_size" : 2000,
        "spectral_hiddens" : [1024, 1024, 512, n_clusters],
        "ae_epochs" : 5,
    }


def autoencode_with_gauss(n_clusters, k):
    return  {
        "n_clusters": n_clusters,
        "should_use_ae" : True,
        "should_use_siamese" : False,
        "ae_hiddens" : [512, 512, 2048, n_clusters],
        "siamese_epochs" : 100,
        "siamese_hiddens" : [1024, 1024, 512, n_clusters],
        "spectral_epochs" : 500,
        "spectral_scale_k" : k,
        "spectral_is_local_scale" : False,
        "spectral_batch_size" : 2000,
        "spectral_hiddens" : [1024, 1024, 512, n_clusters],
        "ae_epochs" : 5,
    }