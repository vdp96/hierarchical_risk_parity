import pandas as pd
import copy


def __check_na_count(df: pd.DataFrame, column: str, row_threshold: float, id_col: str) -> pd.DataFrame:
    """
    Helper function that enforces check on number of rows for each ticker based on input threshold level.
    If count of rows is less than threshold level, omits the ticker from the dataset

    :param df: input dataframe with [date, ticker, value] cols
    :param column: column on which threshold is to be implemented
    :param row_threshold: threshold criteria
    :param id_col: column that identifies the stock [ticker or permno]
    :return:
    """
    # getting ratio of nulls for each ticker in df
    na_count = df.groupby(id_col)[column].apply(lambda x: x.isna().mean())

    # get stocks that need to be omitted
    ban_stocks = na_count[na_count <= row_threshold]

    # final output
    df = df[~df[id_col].isin(ban_stocks)]
    return df


def clean_dataset(df: pd.DataFrame, fillna: bool = True, row_threshold: float = 0.8,
                  id_col: str = "permno") -> pd.DataFrame:
    """
    Utility function that cleans the datasets by checking for:
        1. Nan Values -
            - fill with 0
            - remove those tickers
            - fill with sector mean (TODO)

        2. Duplicates on ticker column
            - drop duplicated tickers

    :param df: input dataframe that needs to be clearned
    :param fillna: Whether to fill NaN values with 0 or drop them
    :param row_threshold: threshold criteria to check if each stock has enough data
    :param id_col: column that identifies the stock [ticker or permno]
    :return: cleaned dataset
    """

    # filtering stocks based on NaN values. At least threshold % should not be NaN's
    df = __check_na_count(df, column="ret", row_threshold=row_threshold, id_col=id_col)

    # taking care of nan values
    if fillna:
        df = df.fillna(0)
    else:
        missing_stocks = df[df.isna().any(axis=1)][id_col].unique()
        df = df[~df[id_col].isin(missing_stocks)]

    # taking care of duplicate tickers
    df = df.drop_duplicates(subset=["date", id_col], keep="last")

    # filtering stocks based on number of data points available. At least threshold % of the max data points per stock
    # should be available
    count_df = df.value_counts(id_col).reset_index()
    max_rows = count_df[0].max()
    req_stocks = count_df[count_df[0] > row_threshold * max_rows][id_col].to_list()
    df = df[df[id_col].isin(req_stocks)]
    return df


def create_correlation_matrix(df: pd.DataFrame, corr_type: str = "spearman", id_col: str = "permno",
                              value_col: str = "ret", covariance: bool = False) -> pd.DataFrame:
    """
    Assuming input df will have the following structure:
    columns:
        1. date
        2. ticker
        3. value

    date        | ticker |  returns
    20220101    AAPL    0.001
    20220102   AAPL     -0.1
    20220101    MSFT    0.14
    20220102    MSFT    -1

    :param df: input returns dataframe
    :param corr_type: type of correlations ["pearson", "spearman"]
    :param id_col: id column in dataframe ["ticker", "permno"]
    :param value_col: column based on which correlations are to be calculates
    :param covariance: whether to calc corr or cov
    :return: correlation df
    """
    df = df.pivot(index="date", columns=id_col, values=value_col)

    if covariance:
        out = df.cov()
    else:
        out = df.corr(method=corr_type)
    return out


def create_return_matrix(df: pd.DataFrame) -> pd.DataFrame:
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
    :return: pivoted dataframe with dates as columns
    ticker        |  20220101 |  20220102
    AAPL            0.001       0.002
    MSFT            0.14        -0.01
    """
    df = df.pivot(index="ticker", columns="date", values="ret")
    return df


def filter_data_for_date_range(df: pd.DataFrame, start_date, end_date, stocks: list = None,
                               id_col: str = "permno") -> pd.DataFrame:
    """
    Helper function to filter dataframe based on input dates, stocks

    :param df: input dataframe
    :param start_date: start date
    :param end_date: end date
    :param stocks: stocks to be filter for if/any
    :param id_col: id column
    :return: filtered dataframe
    """
    if stocks:
        df = df[df[id_col].isin(stocks)]

    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)
    print(f"df.shape: {df.shape}")
    return df


def convert_to_higher_timeframe(daily_df: pd.DataFrame, timeframe: str, date_col: str = "date", id_col: str = "permno",
                                ret_col: str = "ret") -> pd.DataFrame:
    """
    Function to convert daily timeseries returns data to higher aggregated timeseries
    
    :param daily_df: dataframe with daily data
    :param timeframe: weekly, monthly, yearly
    :param date_col: date column
    :param id_col: id column in df
    :param ret_col: returns column
    :return: higher timeseries dataframe
    """
    higher_df = copy.deepcopy(daily_df)
    higher_df[date_col] = pd.to_datetime(higher_df[date_col])

    # set date as index
    higher_df = higher_df.set_index(date_col)

    # getting aggregated returns
    higher_df = higher_df.groupby([id_col, pd.Grouper(freq=timeframe)]).apply(lambda x: (x[ret_col] + 1).prod() - 1)
    higher_df = higher_df.reset_index().rename(columns={0: ret_col})
    return higher_df


def compute_weighted_returns(returns_df: pd.DataFrame, weight_col: str, ret_col: str = "ret") -> pd.DataFrame:
    """
    Calculates weighted returns for input dataframe

    :param returns_df: input returns df
    :param weight_col: weight column
    :param ret_col: returns column
    :return:
    """
    returns_df = copy.deepcopy(returns_df)

    # weighted return column
    weight_ret_col = weight_col + "_" + ret_col

    # adding weighted ret col to df
    returns_df[weight_ret_col] = returns_df["ret"] * returns_df[weight_col]
    return returns_df
