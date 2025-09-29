import numpy as np
from matplotlib import pyplot as plt



class K_Means:
    def __init__(self, k= 5, max_iters= 500, plot_steps= False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample's indices for each cluster
        self.clusters = None

        # the centers (mean vector) for each cluster
        self.centroids = None

    def predict(self, x):
        n_samples, n_features = x.shape

        # initialize centroids
        idxs = np.random.choice(n_samples, self.k, replace= False)
        self.centroids = [x[idx] for idx in idxs]

        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples to closest centroid
            self.clusters = self._get_clusters(self.centroids, x)

            # plot overall
            if self.plot_steps:
                self.plot(x)

            # calculate new centroid from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters, x)

            # check if it has converged
            if self._is_converged(centroids_old, self.centroids):
                break

            # plot overall
            if self.plot_steps:
                self.plot(x)

        # classify samples as the index of their clusters
        labels = np.empty(n_samples)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels


    def _get_clusters(self, centroids, x):
        # assign the samples to the closest centroid
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(x):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [np.sqrt(np.sum((sample - centroid)**2)) for centroid in centroids]
        return np.argmin(distances)

    def _get_centroids(self, clusters, x):
        # assign mean value of each cluster to corresponding centroid
        centroids = [np.mean(x[cluster], axis= 0) for cluster in clusters]
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between old and new centroids, for all centroids
        distances = [np.sqrt(np.sum((old - new)**2)) for old, new in zip(centroids_old, centroids)]
        return sum(distances) == 0

    def plot(self, x):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = x[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()








