import numpy as np
from typing import Tuple


def genrate_data(
    num_samples: int = 100, num_genes: int = 15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random data for supervised clustering (step 1).

    Parameters:
        num_samples (int): Number of samples to generate (default: 100).
        num_genes (int): Number of genes (features) for each sample (default: 15).

    Returns:
        X: A 2D NumPy array of shape (num_samples, num_genes) containing the random data.
        y: A 1D NumPy array of shape (num_samples,) containing binary labels (0 or 1) for each sample.
    """
    X = np.random.randn(num_samples, num_genes)
    y = np.random.randint(0, 2, size=num_samples)
    return X, y


def score(x_1: np.ndarray, x_2: np.ndarray) -> int:
    """
    Compares two 1D NumPy arrays and returns a score.

    Parameters:
      x_1: The first 1D NumPy array.
      x_2: The second 1D NumPy array.

    Returns:
      The score, which is the number of elements in x_1 that are greater than or equal to the corresponding elements in x_2.
    """

    score = 0
    for i in range(x_1.shape[0]):
        score += np.sum(x_1[i] >= x_2)

    return score


def sign_change(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Implement the sign-flip for Pelora (step 2).

    Parameters:
        X (np.ndarray): The data array of shape (num_samples, num_genes).
        y (np.ndarray): The label array of shape (num_samples,).

    Returns:
        X: A 2D NumPy array of shape (num_samples, num_genes) containing the data flip-signe.
        scores: A 1D NumPy array of shape (num_genes,) containing the scores for each gene.
    """
    n, p = X.shape
    _, (n0, n1) = np.unique(y, return_counts=True)

    s_max = n0 * n1  # Maximum possible score

    scores = np.zeros(p)
    new_scores = np.zeros(p)
    for i in range(p):
        xi = X[:, i]
        class_0_values, class_1_values = xi[y == 0], xi[y == 1]
        scores[i] = score(class_0_values, class_1_values)
        if scores[i] > s_max / 2:
            X[:, i] *= -1
        new_scores[i] = min(scores[i], s_max - scores[i])
    return (X, new_scores)


def cluster_intialisation(
    X: np.ndarray, y: np.ndarray, cluster: np.ndarray, scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Intialisation of cluster (step 3).

    Parameters:
        X (2D NumPy array): The data array of shape (num_samples, num_genes).
        y (1D NumPy array): The label array of shape (num_samples,).
        scores (1D NumPy array): The scores for each gene.


    Returns:
        cluster: A 2D NumPy array of shape (num_samples, num_genes) containing the selected gene or genns.
        cluster_index: A 1D NumPy array of shape (num_genes,) containing the index of selected gene or genns.
        initial_cluster_mean: A 1D NumPy array of shape (num_samples,) containing the mean of cluster.
    """
    cluster_index = np.empty((0,))
    margins = np.min(X[y == 1], axis=0) - np.max(X[y == 0], axis=0)

    if not cluster:
        min_value = np.min(scores)
        min_indices = np.where(scores == min_value)[0]

        i_star = (
            np.argmin(scores)
            if min_indices.size == 1
            else np.argmax(margins[min_indices])
        )
        cluster = np.expand_dims(X[:, i_star], axis=1)
        cluster_index = np.append(cluster_index, i_star)
        initial_cluster_mean = X[:, i_star]
        return cluster, cluster_index, initial_cluster_mean
    else:
        initial_cluster_mean = np.mean(cluster, axis=1)
        return cluster, np.empty((0,)), initial_cluster_mean


def forward_search(
    X: np.ndarray,
    y: np.ndarray,
    cluster: np.ndarray,
    cluster_index: np.ndarray,
    initial_cluster_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward Search (step 4-5).

    Parameters:
        X (2D NumPy array): The data array of shape (num_samples, num_genes).
        y (1D NumPy array): The label array of shape (num_samples,).
        cluster (2D NumPy array): The selected gene or genns.
        cluster_index (1D NumPy array): The index of selected gene or genns.
        initial_cluster_mean (1D NumPy array): The mean of cluster.

    Returns:
        cluster: A 2D NumPy array of shape (num_samples, num_genes) containing the clustered genes.
        cluster_index: A 1D NumPy array of shape (num_genes,) containing the index of clustered genes.
    """

    n, p = X.shape
    _, (n0, n1) = np.unique(y, return_counts=True)

    s_max = n0 * n1  # Maximum possible score
    cluster_mean_score = score(
        initial_cluster_mean[y == 0], initial_cluster_mean[y == 1]
    )
    cluster_mean_margin = np.min(initial_cluster_mean[y == 1], axis=0) - np.max(
        initial_cluster_mean[y == 0], axis=0
    )

    while True:
        scores_with_gene = np.zeros(p)
        margins_with_gene = np.zeros(p)
        for i in range(p):
            if i in cluster_index:
                # s_max+1 is the big score for to not be selected as i_star
                scores_with_gene[i] = s_max + 1
                # impossible to found gene with margin 0 for to not be selected as i_star
                margins_with_gene[i] = 0
                continue
            temp_cluster = np.concatenate(
                [cluster, np.expand_dims(X[:, i], axis=1)], axis=1
            )
            temp_cluster_avg = np.mean(temp_cluster, axis=1)
            scores_with_gene[i] = score(
                temp_cluster_avg[y == 0], temp_cluster_avg[y == 1]
            )
            margins_with_gene[i] = np.min(temp_cluster_avg[y == 1]) - np.max(
                temp_cluster_avg[y == 0]
            )

        min_value = np.min(scores_with_gene)
        min_indices = np.where(scores_with_gene == min_value)[0]

        i_star = (
            min_indices[0]
            if min_indices.size == 1
            else np.argmax(margins_with_gene[min_indices])
        )

        if scores_with_gene[i_star] >= cluster_mean_score:
            break

        if scores_with_gene[i_star] == cluster_mean_score:
            if margins_with_gene[i_star] <= cluster_mean_margin:
                break

        cluster_index = np.append(cluster_index, i_star)
        cluster = np.concatenate(
            [cluster, np.expand_dims(X[:, i_star], axis=1)], axis=1
        )
        cluster_mean_score = scores_with_gene[i_star]
        cluster_mean_margin = margins_with_gene[i_star]
    return cluster, cluster_index


def backword_search(
    y: np.ndarray,
    cluster: np.ndarray,
    cluster_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backword Search (step 6-7).

    Parameters:
        y (1D NumPy array): The label array of shape (num_samples,).
        cluster (2D NumPy array): The selected gene or genns.
        cluster_index (1D NumPy array): The index of selected gene or genns.

    Returns:
        cluster: A 2D NumPy array of shape (num_samples, num_genes) containing the clustered genes.
        cluster_index: A 1D NumPy array of shape (num_genes,) containing the index of clustered genes.
        initial_cluster_mean: A 1D NumPy array of shape (num_samples,) containing the mean of cluster.
    """

    initial_cluster_mean = np.mean(cluster, axis=1)
    cluster_mean_score = score(
        initial_cluster_mean[y == 0], initial_cluster_mean[y == 1]
    )
    cluster_mean_margin = np.min(initial_cluster_mean[y == 1], axis=0) - np.max(
        initial_cluster_mean[y == 0], axis=0
    )

    while True:
        scores_without_gene = np.zeros(cluster_index.size)
        margins_without_gene = np.zeros(cluster_index.size)
        # start from last gene
        for i in range(-1, -cluster_index.size - 1, -1):
            cluster_mean_without_gene_i = np.mean(np.delete(cluster, i, axis=1), axis=1)
            scores_without_gene[i] = score(
                cluster_mean_without_gene_i[y == 0], cluster_mean_without_gene_i[y == 1]
            )
            margins_without_gene[i] = np.min(
                cluster_mean_without_gene_i[y == 1], axis=0
            ) - np.max(cluster_mean_without_gene_i[y == 0], axis=0)

        min_value = np.min(scores_without_gene)
        min_indices = np.where(scores_without_gene == min_value)[0]

        i_star = (
            min_indices[0]
            if min_indices.size == 1
            else np.argmax(margins_without_gene[min_indices])
        )
        if scores_without_gene[i_star] > cluster_mean_score:
            break

        if scores_without_gene[i_star] == cluster_mean_score:
            if margins_without_gene[i_star] <= cluster_mean_margin:
                break

        cluster = np.delete(cluster, i_star, axis=1)
        cluster_index = np.delete(cluster_index, i_star, axis=0)
        cluster_mean_score = scores_without_gene[i_star]
        cluster_mean_margin = margins_without_gene[i_star]
    initial_cluster_mean = np.mean(cluster, axis=1)
    return cluster, cluster_index, initial_cluster_mean


if __name__ == "__main__":
    n, p = (100, 15)
    X, y = genrate_data(num_samples=n, num_genes=p)
    X, scores = sign_change(X, y)
    cluster, cluster_index, intiale_cluster_mean = cluster_intialisation(
        X=X, y=y, cluster=None, scores=score
    )
    cluster, cluster_index = forward_search(
        X, y, cluster, cluster_index, intiale_cluster_mean
    )
    print(cluster_index)
    cluster, cluster_index, intiale_cluster_mean = backword_search(
        y, cluster, cluster_index
    )
    print(cluster_index)
