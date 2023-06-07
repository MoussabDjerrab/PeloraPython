import numpy as np
from typing import Tuple
from pelora import *


def supervides_clustering(
    X: np.ndarray, y: np.ndarray, initial_cluster: np.ndarray, noc: int
):
    """
    Supervised Clustering PELORA.

    Parameters:
        X (2D NumPy array): The data array of shape (num_samples, num_genes).
        y (1D NumPy array): The label array of shape (num_samples,).
        initial_cluster (2D NumPy array): The intiale cluster.
        noc (int): Number of cluster.


    Returns:
        clusters: A 2D NumPy array containing all cluster.
    """
    clusters = []
    X, scores = sign_change(X, y)
    for i in range(noc):
        cluster, cluster_index, initial_cluster_mean = cluster_intialisation(
            X=X, y=y, cluster=initial_cluster, scores=scores
        )
        cluster, cluster_index = forward_search(
            X, y, cluster, cluster_index, initial_cluster_mean
        )
        cluster, cluster_index, initial_cluster_mean = backword_search(
            y, cluster, cluster_index
        )
        cluster_index_state = []
        cluster_index_state.append(cluster_index)

        while True:
            cluster, cluster_index = forward_search(
                X, y, cluster, cluster_index, initial_cluster_mean
            )
            cluster, cluster_index, initial_cluster_mean = backword_search(
                y, cluster, cluster_index
            )
            cluster_index_state.append(cluster_index)

            if np.array_equal(cluster_index_state[-2], cluster_index_state[-1]):
                break
        clusters.append(cluster_index)
        X = np.delete(X, cluster_index.astype(int), axis=1)
        scores = np.delete(scores, cluster_index.astype(int), axis=0)
    return clusters


if __name__ == "__main__":
    n, p = (100, 15)
    X, y = genrate_data(num_samples=n, num_genes=p)

    cluster, cluster_index, intiale_cluster_mean = cluster_intialisation(
        X=X, y=y, cluster=None, scores=score
    )
    cluster_index = supervides_clustering(X=X, y=y, initial_cluster=None, noc=3)
    print(cluster_index)
