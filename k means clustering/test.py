import numpy as np
from k_means import K_Means
from sklearn.datasets import make_blobs


# Testing
if __name__ == "__main__":
    np.random.seed(42)

    x, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    print(x.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = K_Means(k=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(x)

    k.plot(x)
