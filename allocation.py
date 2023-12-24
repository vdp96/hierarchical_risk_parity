import pandas as pd
import numpy as np
import hrp
import utils as hrp_utils


def compute_hrp_weights(df: pd.DataFrame, linkage_type: str = "single", distance_type: str = "euclidean") -> pd.Series:
    """
    Computes HRP allocation for a portfolio according to initial paper. Has additional linkage types to explore

    :param df: dataframe with assets, returns and date
    :param linkage_type: [single, complete, ward, average]
    :param distance_type: [euclidean,
    :return:
    """

    # correlation and covariance matrix
    correlation_matrix = hrp_utils.create_correlation_matrix(df=df, corr_type="spearman")
    covariance_matrix = hrp_utils.create_correlation_matrix(df=df, covariance=True)

    # linkage matrix
    linkage_matrix = hrp.get_linkage(corr_matrix=correlation_matrix, linkage_type=linkage_type,
                                     distance_type=distance_type)

    # quasi diagonalization
    qd_idx = hrp.perform_quasi_diagonalization(linkage_matrix=linkage_matrix)

    # weights from recursive bipartition
    weights = hrp.perform_recursive_bipartition(covariance_matrix=covariance_matrix, quasi_diag_idx=qd_idx)

    # creating output series with tickers and weights
    stocks = covariance_matrix.columns[qd_idx]
    weights.index = stocks
    weights.name = "HRP" + "_" + linkage_type + "_wgt"
    return weights


def compute_mv_weights(df: pd.DataFrame) -> pd.Series:
    """
    Computes weights based on minimum variance allocation
    :param df: dataframe with assets, returns and date
    :return: Series with tickers and weights
    """
    # covariance matrix
    covariance_matrix = hrp_utils.create_correlation_matrix(df=df, covariance=True)
    stocks = covariance_matrix.columns

    # inverse cov matrix
    inv_covar = np.linalg.inv(covariance_matrix)

    # calculating min var allocation
    u = np.ones(len(covariance_matrix))
    weights = np.dot(inv_covar, u) / np.dot(u, np.dot(inv_covar, u))
    return pd.Series(weights, index=stocks, name="MV")


def compute_rp_weights(df: pd.DataFrame) -> pd.Series:
    """
    Computes Risk parity allocation

    :param df: dataframe with assets, returns and date
    :return: Series with tickers and weights
    """
    # covariance matrix
    covariance_matrix = hrp_utils.create_correlation_matrix(df=df, covariance=True)
    stocks = covariance_matrix.columns

    # risk parity allocation
    weights = (1 / np.diag(covariance_matrix))
    weights /= sum(weights)
    return pd.Series(weights, index=stocks, name="RP")


def compute_unif_weights(df: pd.DataFrame, id_col: str = "permno") -> pd.Series:
    """
    Computes equal weights

    :param df: dataframe with assets, returns and date
    :param id_col: identifier column
    :return: Series with tickers and weights
    """

    df["EQ"] = 1 / len(df)
    df = df.set_index(id_col)
    weights = df["EQ"].squeeze()

    return weights


def compute_mkt_weights(df: pd.DataFrame, mkt_col: str = "lag_mkt_cap", id_col: str = "permno") -> pd.Series:
    """
    Computes market weights

    :param df: dataframe with assets, market weight
    :param mkt_col: market cap column based on which weights are to be calculates
    :param id_col: identifier column
    :return: Series with tickers and weights
    """

    df["MKT"] = df[mkt_col] / df[mkt_col].sum()
    df = df.set_index(id_col)
    weights = df["MKT"].squeeze()

    return weights
