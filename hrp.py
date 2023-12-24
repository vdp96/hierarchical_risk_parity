import pandas as pd
import numpy as np
import utils as hrp_utils
import os
from scipy.cluster.hierarchy import linkage, leaves_list


def get_linkage(corr_matrix: np.array, linkage_type: str, distance_type: str) -> np.array:
    """
    This function gets linkage matrix using scipy hierarchy library

    :param corr_matrix: correlation matrix for the universe
    :param linkage_type: ["single", "ward", "complete", "average"]
    :param distance_type:
    :return: linkage matrix
    """
    # distance
    corr_distances = np.sqrt(0.5 * (1 - corr_matrix))

    # linkage
    linkage_matrix = linkage(y=corr_distances, method=linkage_type, metric=distance_type)
    return linkage_matrix


def perform_quasi_diagonalization(linkage_matrix: np.array) -> np.array:
    """
    Quasi-Diagonalize the linkage matrix i.e. rearrange the linkage matrix so that highly-correlated stock lie along
    the diagonal

    :param linkage_matrix:
    :return: quasi diagonalised matrix
    """
    sorted_ix = leaves_list(linkage_matrix)
    return sorted_ix


def perform_recursive_bipartition(covariance_matrix: pd.DataFrame, quasi_diag_idx: np.array) -> pd.Series:
    """
    Computes HRP allocation:
    This function performs recursive bisection and assigns allocation according to HRP paper

    :param covariance_matrix: input covariance matrix for all stocks
    :param quasi_diag_idx: quasi diagonalisation sorted indices list
    :return: weights for each stock
    """
    # initialising weights to 1 for all stock
    weights = pd.Series(1, index=quasi_diag_idx)

    # start with top-down partition beginning with one cluster
    cluster_items = [quasi_diag_idx]

    while len(cluster_items) > 0:
        # bisection
        items = []

        # divide each cluster into 2 parts recursively
        for cluster in cluster_items:
            if len(cluster) > 1:
                c1 = cluster[:len(cluster) // 2]
                c2 = cluster[len(cluster) // 2:]
                items.append(c1)
                items.append(c2)
        cluster_items = items

        # assign weights for each of the 2 clusters
        for i in range(0, len(cluster_items), 2):
            cluster1 = cluster_items[i]
            cluster2 = cluster_items[i + 1]

            # getting cluster variance according to wT.V.w for each cluster
            cluster_var1 = get_cluster_variance(covariance_matrix, cluster1)
            cluster_var2 = get_cluster_variance(covariance_matrix, cluster2)

            # HRP allocation alpha (equivalent to inverse vol)
            alpha = 1 - cluster_var1 / (cluster_var1 + cluster_var2)
            weights[cluster1] *= alpha
            weights[cluster2] *= 1 - alpha
    return weights


def get_cluster_variance(covariance_matrix: pd.DataFrame, cluster_idx: np.array) -> float:
    """
    Computes cluster variance based on inverse vol of stock within the cluster

    :param covariance_matrix: covariance matrix with all stocks
    :param cluster_idx: indexes of stocks within cluster
    :return: cluster variance
    """
    cov = covariance_matrix.iloc[cluster_idx, cluster_idx]  # matrix slice

    # calculate the inverse-variance portfolio
    inv_var = 1. / np.diag(cov)
    inv_var /= inv_var.sum()
    w = inv_var.reshape(-1, 1)
    cluster_var = np.dot(np.dot(w.T, cov), w)[0, 0]
    return cluster_var


def do():
    # directory information
    CUR_DIR = os.path.dirname(os.getcwd())
    DATA_FOLDER = os.path.join(CUR_DIR, "data")
    ALL_DATA = os.path.join(DATA_FOLDER, "all_data.pkl")
    data = pd.read_pickle(ALL_DATA)

    data = hrp_utils.filter_data_for_date_range(df=data, start_date="2021-01-01", end_date="2022-01-01",
                                                id_col="permno")
    data = data[["date", "ticker", "ret"]]

    data = hrp_utils.clean_dataset(df=data, fillna=True)
    corr = hrp_utils.create_correlation_matrix(df=data)
    link = get_linkage(corr_matrix=corr, linkage_type="ward", distance_type="euclidean")
    x = perform_quasi_diagonalization(link)
    cov = hrp_utils.create_correlation_matrix(data, covariance=True)

    w = perform_recursive_bipartition(covariance_matrix=cov, quasi_diag_idx=x)

    return w

# print(do())
