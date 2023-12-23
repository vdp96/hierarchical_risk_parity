import pandas as pd
import numpy as np
import wrds_loader
import os
import copy
import json
from sklearn.cluster import AgglomerativeClustering

# directory information
PATH = os.path.dirname(os.getcwd())
DATA_FOLDER = os.path.join(PATH, "data")
RETURNS_FOLDER = os.path.join(DATA_FOLDER, "returns")
ALL_DATA = "/Users/vdp/mfe/quarters/quarter_4_fall_2023/AFP/data/processed_data/all_data.pkl"


def check_na_count(df, column, row_threshold=0.8):
    # getting ratio of nulls for each ticker in df
    na_count = df.groupby("ticker")[column].apply(lambda x: x.isna().mean())

    ban_tickers = na_count[na_count <= row_threshold]
    df = df[~df.ticker.isin(ban_tickers)]
    return df


def clean_dataset(df, fillna=True, row_threshold=0.8):
    """
    Function check for:
        1. Nan Values -
            - fill with 0
            - remove those tickers
            - fill with average or some ffill

        2. Duplicates on ticker column
            - drop duplicated tickers
    :param df:
    :return:
    """
    df = check_na_count(df, column="ret", row_threshold=row_threshold)

    # Take care of nan values
    if fillna:
        df = df.fillna(0)
        # check na count
    else:
        missing_tickers = df[df.isna().any(axis=1)]["ticker"].unique()
        df = df[~df["ticker"].isin(missing_tickers)]

    # Take care of duplicate tickers
    df = df.drop_duplicates(subset=["date", "ticker"], keep="first")

    count_df = df.value_counts("ticker").reset_index()
    max_rows = count_df[0].max()
    tickers = count_df[count_df[0] > row_threshold * max_rows]["ticker"].to_list()
    df = df[df["ticker"].isin(tickers)]
    return df


def create_correlation_matrix(df, corr_type="spearman", covariance=False):
    """
    Assuming input df will have the following structure:
    columns:
        1. date
        2. ticker
        3. return

    date        | ticker |  returns
    20220101    AAPL    0.001
    20220102   AAPL     -0.1
    20220101    MSFT    0.14
    20220102    MSFT    -1

    :param df:
    :param corr_type:
    :param covariance:
    :return: correlation df
    """
    df = df.pivot(index="date", columns="ticker", values="ret")

    if covariance:
        out = df.cov()
    else:
        out = df.corr(method=corr_type)
    return out


def create_return_matrix(df):
    """
    Assuming input df will have the following structure:
    columns:
        1. date
        2. ticker
        3. return

    date        | ticker |  returns
    20220101    AAPL    0.001
    20220102   AAPL     -0.1
    20220101    MSFT    0.14
    20220102    MSFT    -1

    :param df:
    :return:
    ticker        |  20220101 |  20220102
    AAPL            0.001       0.002
    MSFT            0.14        -0.01
    """
    df = df.pivot(index="ticker", columns="date", values="ret")
    return df


def get_snp_constituents_for_date(date, wrds_id):
    snp_data = wrds_loader.download_snp_constituents(wrds_id=wrds_id)
    snp_for_date = snp_data[snp_data["date"] == date]

    # tickers = snp_for_date["ticker"].tolist()
    ticker_name_map = snp_for_date.set_index("ticker")["comnam"].to_dict()
    return ticker_name_map


def __dump_snp_constituents_all_years():
    snp_df = wrds_loader.download_snp_constituents(wrds_id="vpunjala1996")
    snp_df["date"] = pd.to_datetime(snp_df.date)
    max_dates = snp_df.groupby(pd.Grouper(key="date", freq="1Y"))["date"].max().to_list()

    snp_constituents = dict()

    for dt in max_dates:
        tickers = snp_df[snp_df["date"] == dt].ticker.to_list()
        snp_constituents[str(dt.year + 1)] = tickers

    snp_file = os.path.join(DATA_FOLDER, "snp_constituents.json")
    with open(snp_file, "w") as fp:
        json.dump(snp_constituents, fp)
    return


def __dump_data_to_feather_file():
    returns_files = os.listdir(RETURNS_FOLDER)

    # Read ticker data
    df_list = list()
    for file in returns_files:
        file_path = os.path.join(RETURNS_FOLDER, file)
        fdf = pd.read_feather(file_path)
        df_list.append(fdf)

    df = pd.concat(df_list, axis=0, ignore_index=True)
    file_name = "all_file.feather"
    file_path = os.path.join(RETURNS_FOLDER, file_name)
    df.to_feather(file_path)
    return


def get_data_for_date_range(tickers, start_date, end_date):
    # all_file_path = os.path.join(DATA_FOLDER, "all_data.pkl")
    df = pd.read_pickle(ALL_DATA)

    df = df[df["ticker"].isin(tickers)]
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)
    print(f"df.shape: {df.shape}")
    return df


def get_gics_data(date: str = None):
    gics_file = os.path.join(DATA_FOLDER, "GICS.csv")
    df = pd.read_csv(gics_file)
    req_cols = ["datadate", "tic", "Sector", "IndustryGroup", "Industry", "SubIndustry"]

    df = df[req_cols]
    df.columns = ["date", "ticker", "sector", "industry_group", "industry", "sub_industry"]

    if date:
        df = df[df["date"] == date].reset_index(drop=True)
    print(f"df.shape: {df.shape}")
    return df


def create_linkage_matrix(clustering):
    # create the counts of samples under each node
    counts = np.zeros(clustering.children_.shape[0])
    n_samples = len(clustering.labels_)

    for i, merge in enumerate(clustering.children_):
        current_count = 0

        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, counts]).astype(float)

    return linkage_matrix


def get_distance_matrix(corr_matrix):
    return np.sqrt((1 - corr_matrix) / 2)


def perform_clustering(df, cluster_list):
    corr_df = create_correlation_matrix(df, "spearman")
    dist_df = get_distance_matrix(corr_df)

    for n in cluster_list:
        clusters = AgglomerativeClustering(n_clusters=n, linkage="ward").fit_predict(dist_df)
        cluster_map = dict(zip(dist_df.columns, clusters))
        df[f"cluster_{n}"] = df["ticker"].map(cluster_map)
    return df


def get_correlation_summary(cl_df):
    cl_df = create_correlation_matrix(cl_df)
    cl_df = cl_df.stack()
    cl_df.index.names = ["ticker1", "ticker2"]
    cl_df = cl_df.reset_index()
    cl_df = cl_df[cl_df["ticker1"] != cl_df["ticker2"]]

    mean_corr = cl_df[0].mean()
    return mean_corr


def get_cluster_correlation_summary(cluster_df, cluster_columns):
    cluster_wise_corr_dict = dict()
    year_wise_corr_dict = dict()

    for cluster_col in cluster_columns:
        corr_df = cluster_df.groupby(["year", cluster_col]).apply(
            lambda x: get_correlation_summary(x)).reset_index().rename(columns={0: "correlation"})
        cluster_wise_corr_dict[cluster_col] = corr_df[["year", cluster_col, "correlation"]]
        year_wise_corr_dict[cluster_col] = corr_df.reset_index().groupby("year")[
            "correlation"].mean().reset_index().rename(columns={0: "correlation"})

    return cluster_wise_corr_dict, year_wise_corr_dict


def get_monthly_returns(daily_df):
    monthly_df = copy.deepcopy(daily_df)
    monthly_df["date"] = pd.to_datetime(monthly_df["date"])

    # Set date as index
    monthly_df = monthly_df.set_index("date")

    # Getting Monthly Returns
    monthly_df = monthly_df.groupby(["ticker", pd.Grouper(freq="M")]).apply(lambda x: (x["ret"] + 1).prod() - 1)
    monthly_df = monthly_df.reset_index().rename(columns={0: "ret"})
    return monthly_df


def get_yearly_returns(monthly_df, weight_col=None):
    monthly_df = copy.deepcopy(monthly_df)

    if weight_col:
        weight_ret_col = weight_col + "_ret"
        monthly_df[weight_ret_col] = monthly_df["ret"] * monthly_df[weight_col]
    else:
        weight_ret_col = "ret"

    monthly_df["date"] = pd.to_datetime(monthly_df["date"])

    # Set date as index
    monthly_df = monthly_df.set_index("date")

    # Getting Monthly Returns
    monthly_df = monthly_df.groupby(["ticker", pd.Grouper(freq="Y")]).apply(
        lambda x: (x[weight_ret_col] + 1).prod() - 1)
    monthly_df = monthly_df.reset_index().rename(columns={0: weight_ret_col})
    return monthly_df
