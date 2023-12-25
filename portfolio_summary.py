import numpy as np
import pandas as pd


def __compute_cum_ret(ret_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that calculates Compounded Annual Growth Rate

    :param ret_df: input data frame with daily return series (date as index)
    :return: returns summary df with index as return type, 1 CAGR column
    """

    cum_ret = (((ret_df + 1).prod()) - 1).to_frame().rename(columns={0: "CUM_RET (%)"}) * 100
    return cum_ret


def __compute_cagr(ret_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that calculates Compounded Annual Growth Rate

    :param ret_df: input data frame with daily return series (date as index)
    :return: returns summary df with index as return type, 1 CAGR column
    """

    years = len(pd.to_datetime(ret_df.index).year.unique())
    cum_ret = (((ret_df + 1).prod()) ** (1 / years) - 1).to_frame().rename(columns={0: "CAGR (%)"}) * 100
    return cum_ret


def __compute_mean_return(ret_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that calculates Mean Annual Return

    :param ret_df: input data frame with daily return series (date as index)
    :return: returns summary df with index as return type, 1 Mean Ret column
    """

    mean_ret = ret_df.groupby(pd.Grouper(freq="Y")).apply(lambda x: (1 + x).prod() - 1).mean().to_frame().rename(
        columns={0: "MAR (%)"}) * 100
    return mean_ret


def __compute_annual_volatility(ret_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that calculates Annual Volatility

    :param ret_df: input data frame with daily return series (date as index)
    :return: returns summary df with index as return type, 1 vol column
    """

    ann_vol = (ret_df.std() * np.sqrt(252)).to_frame().rename(columns={0: "ANN_VOL (%)"}) * 100
    return ann_vol


def __compute_max_drawdown(ret_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that calculates Annual Volatility

    :param ret_df: input data frame with daily return series (date as index)
    :return: returns summary df with index as return type, 1 vol column
    """

    cum_ret = (1 + ret_df).cumprod()
    max_ret = cum_ret.cummax()
    drawdown = (cum_ret - max_ret) / max_ret

    max_drawdown = (drawdown.min()).to_frame().rename(columns={0: "MAX_DD (%)"}) * 100
    return max_drawdown


def ___compute_sharpe_ratio(ret_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that calculates Annual Volatility

    :param ret_df: input data frame with daily return series (date as index)
    :return: returns summary df with index as return type, 1 sharpe ratio column
    """

    ret_df = ret_df.groupby(pd.Grouper(freq="Y")).apply(lambda x: (1 + x).prod() - 1)
    sharpe_df = (ret_df.mean() / ret_df.std()).to_frame().rename(columns={0: "SHARPE"})
    return sharpe_df


def __compute_t_stat(ret_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function that calculates Annual Volatility

    :param ret_df: input data frame with daily return series (date as index)
    :return: returns summary df with index as return type, 1 t-stat column
    """

    ret_df = ret_df.groupby(pd.Grouper(freq="Y")).apply(lambda x: (1 + x).prod() - 1)
    tstat_df = ((ret_df.mean() / ret_df.std()) * np.sqrt(len(ret_df))).to_frame().rename(columns={0: "TSTAT"})
    return tstat_df


def compute_portfolio_summary(ret_df: pd.DataFrame, return_columns: list = None) -> pd.DataFrame:
    """
    Wrapper function that calculates portfolio summary statistics

    :param ret_df: input return timeseries with dates, returns
    :param return_columns: optional if any
    :return: portfolio summary
    """
    if return_columns:
        ret_df = ret_df[return_columns]

    cum_ret = __compute_cum_ret(ret_df=ret_df)
    cagr = __compute_cagr(ret_df=ret_df)
    mar = __compute_mean_return(ret_df=ret_df)
    ann_vol = __compute_annual_volatility(ret_df=ret_df)
    maxdd = __compute_max_drawdown(ret_df=ret_df)
    sharpe = ___compute_sharpe_ratio(ret_df=ret_df)
    tstat = __compute_t_stat(ret_df=ret_df)

    summary_df = pd.concat([cum_ret, cagr, mar, ann_vol, sharpe, tstat, maxdd], axis=1).round(4)

    return summary_df
