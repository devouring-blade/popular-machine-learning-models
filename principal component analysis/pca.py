import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, x):
        # mean centering
        self.mean = np.mean(x, axis= 0)
        x = x - self.mean

        # covariance
        cov = np.cov(x.T)

        # eigenvectors, eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # sort eigenvectors according eigenvalues
        idxs = np.argsort(eigenvalues, )[: : -1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[: , idxs]

        self.components = eigenvectors[: , : self.n_components]

    def transform(self, x):
        # centering data
        x = x - self.mean
        return x @ self.components






