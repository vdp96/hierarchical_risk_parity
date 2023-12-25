# Portfolio Construction with Hierarchical Risk Parity - A Study
The goal of project is to study the famous portfolio construction technique (HRP) developed by Prof. Marcos López de Prado(2016), build up on it by exploring various linkages, and perform a comparitive analysis of HRP portfolios against base cases.  

The original paper developed by Prof. Marcos López de Prado(2016) uses "Single" linkage in building HRP portfolio. Single linkages uses **minimum distance criteria** to cluster the stocks together. However, there are various other methods such as "Complete", "Ward", and "Average" methods available to cluster the stocks which have not been explored in the literature.  

In this project, I explored 4 different linkages (single, complete, ward, average) within hierarchical clustering to see if there is any significant difference in portfolio allocation and out of sample performance. Furthermore, I performed portfolio analysis and calculated summary statiscs of various HRP portfolio and benchmarked it against various base cases.       

**Link to project:** https://github.com/vdp96/hierarchical_risk_parity/blob/main/performance_study.ipynb


## How It's Made:

**Tech Used:** Python, Jupyter Notebook

**Universe:** SNP500

**Timeframe:** 2001 - 2022

**Data Source:** WRDS Database

All the data used within this project has been acquired from WRDS Database. Since the purpose of the project is to explore HRP and data acquisition, I just used a processed data set saved on my local folder.

**Main Ideas and Assumptions:** 
- Returns are in "daily" timeframe
- SNP500 constituents are rebalanced every year. That is, SNP500 constituents are acquired at the beginning of each year and remain same for the entire year
- **Cleaned and Processed Dataset:** Stocks containing >20% NaN data points in a year are dropped. Each stock has atleast 80% of the data in a year else it is dropped. Duplicates stocks are dropped from the dataset. Rest of the NaN values are filled with 0 for the purpose of this study.
- HRP weights for current year are derived from HRP algorithm using correlation matrix, which formed using previous year returns 
- Similarly, for market weighting, last available market cap of each stock in previous year (lagged market cap) is used
- Similarly, for risk parity weighting, previous year stock volatilities are used

## Lessons Learned:

1. Changing linkage types in HRP portfolios did not produce significant difference in portfolio allocation and performance
2. HRP portfolios had lower annual volatility compared to base cases
3. HRP portfolios had lower max drawdown compared to base cases
4. HRP portfolios outperformed the base cases of Market, Equal and Risk Parity Weighting in terms of sharpe ratio (return over risk)
5. Overall, it looks like HRP is indeed a superior portfolio construction technique for investors looking to attain higher risk over return
![img.png](img.png)