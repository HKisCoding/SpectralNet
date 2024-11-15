class ExperimentConfig():
    def __init__(self, n_clusters, k):
        self.n_clusters = n_clusters
        self.k = k
        self.net_layers = {
            "ae_hiddens": [64, 64, 32, n_clusters],
            "siamese_hiddens" : [256, 256, 64, n_clusters],
            "spectral_hiddens" : [512, 512, 64, n_clusters],
        }
    def autoencoder_with_siamese(self):
        return  {
            "n_clusters": self.n_clusters,
            "should_use_ae" : True,
            "should_use_siamese" : True,
            "siamese_epochs" : 100,
            "spectral_epochs" : 500,
            # spectral_scale_k= 20,
            "spectral_is_local_scale" : True,
            "spectral_batch_size" : 2000,
            "ae_epochs" : 5,
            **self.net_layers
        }


    def autoencode_with_gauss_all_data(self):
        return  {
            "n_clusters": self.n_clusters,
            "should_use_ae" : True,
            "should_use_siamese" : False,
            "siamese_epochs" : 100,
            "spectral_epochs" : 500,
            "spectral_is_local_scale" : True,
            "spectral_batch_size" : 2000,
            "ae_epochs" : 5,
        }


    def autoencode_with_gauss_k_neighbor(self, n_clusters):
        return  {
            "n_clusters": self.n_clusters,
            "should_use_ae" : True,
            "should_use_siamese" : False,
            "siamese_epochs" : 100,
            "spectral_epochs" : 500,
            "spectral_scale_k" : self.k,
            "spectral_is_local_scale" : False,
            "spectral_batch_size" : 2000,
            "ae_epochs" : 5,
        }