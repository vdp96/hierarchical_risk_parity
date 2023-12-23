import wrds
import pandas as pd
import datetime
import os


def download_stock_data_crsp(ticker_list, chunk_size=100, conn=None, wrds_id=None):
    conn_flag = False

    # Connect to WRDS 
    if not conn:
        conn_flag = True
        conn = wrds.Connection(wrds_username=wrds_id)

    stock_df = pd.DataFrame()

    # Split the ticker list into chunks
    ticker_chunks = [ticker_list[i:i + chunk_size] for i in range(0, len(ticker_list), chunk_size)]

    for chunk in ticker_chunks:
        # Convert each chunk of tickers to a format suitable for SQL IN clause
        ticker_str = ",".join(f"'{ticker}'" for ticker in chunk)

        # Query the CRSP database for daily returns
        chunk_df = conn.raw_sql(f"""
            SELECT a.permno, a.date, a.ret, a.shrout, a.prc, a.bid, a.ask, a.numtrd, b.ticker
            FROM crsp.dsf as a
            JOIN crsp.dsenames AS b
            ON a.permno = b.permno
            WHERE b.ticker IN ({ticker_str})
            AND b.namedt <= a.date
            AND (b.nameendt >= a.date OR b.nameendt IS NULL)
            ORDER BY a.date;
        """, date_cols=["date"])

        # Convert negative prices to positive
        chunk_df["prc"] = chunk_df["prc"].abs()

        # Append the chunk data to the main dataframe
        stock_df = pd.concat([stock_df, chunk_df], ignore_index=True)

    if conn_flag:
        conn.close()

    return stock_df


def download_snp_constituents(daily_flag=False, wrds_id=None):
    conn = wrds.Connection(wrds_username=wrds_id)

    p = "m"
    if daily_flag:
        p = "d"

    sp500 = conn.raw_sql(f"""
                        select a.*, b.date, b.ret
                        from crsp.{p}sp500list as a,
                        crsp.{p}sf as b
                        where a.permno=b.permno
                        and b.date >= a.start and b.date<= a.ending
                        and b.date>='01/01/2000'
                        order by date;
                        """, date_cols=["start", "ending", "date"])

    mse = conn.raw_sql("""
                        select comnam, ncusip, namedt, nameendt, 
                        permno, shrcd, exchcd, hsiccd, ticker
                        from crsp.msenames
                        """, date_cols=["namedt", "nameendt"])

    # if nameendt is missing then set to today date
    mse["nameendt"] = mse["nameendt"].fillna(pd.to_datetime("today"))

    # Merge with SP500 data
    sp500_full = pd.merge(sp500, mse, how="left", on="permno")

    # Impose the date range restrictions
    sp500_full = sp500_full.loc[(sp500_full.date >= sp500_full.namedt) \
                                & (sp500_full.date <= sp500_full.nameendt)]

    # Linking with Compustat through CCM
    ccm = conn.raw_sql("""
                      select gvkey, liid as iid, lpermno as permno, linktype, linkprim, 
                      linkdt, linkenddt
                      from crsp.ccmxpf_linktable
                      where substr(linktype,1,1)='L'
                      and (linkprim ='C' or linkprim='P')
                      """, date_cols=["linkdt", "linkenddt"])

    # if linkenddt is missing then set to today date
    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.to_datetime("today"))

    # Merge the CCM data with S&P500 data
    # First just link by matching PERMNO
    sp500ccm = pd.merge(sp500_full, ccm, how="left", on=["permno"])

    # Then set link date bounds
    sp500ccm = sp500ccm.loc[(sp500ccm["date"] >= sp500ccm["linkdt"]) \
                            & (sp500ccm["date"] <= sp500ccm["linkenddt"])]

    return sp500ccm


def download_finratios(ticker, db=None):
    none_flag = False
    if not db:
        none_flag = True
        db = wrds.Connection()

    data = db.raw_sql(
        f"""SELECT 
                ticker,
                public_date as date,
                pe_op_basic,
                pe_exi,
                pe_inc, 
                bm 
            FROM
                wrdsapps_finratio.firm_ratio
            WHERE ticker = '{ticker}'""")
    data["date"] = pd.to_datetime(data["date"])
    if none_flag:
        db.close()
    return data[["ticker", "date", "pe_op_basic", "pe_exi", "pe_inc", "bm"]].dropna()