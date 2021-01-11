import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import argparse
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")


def portfolio_vol(w, cov):
    """
    Computes the portfolio risk
    """
    return (w.T @ cov @ w)


def risk_contribution(w, cov):
    """
    Computes the risk contributions of the different portfolio constituents, provided the portfolio weights and the covariance matrix
    """
    return np.multiply(cov @ w.T, w.T)/portfolio_vol(w,cov)


def optim_func(w, cov):
    """
    provides the mean squared difference between the risk contributions and targeted risk
    """
    n = cov.shape[0]
    target_risk = np.repeat(1/n, n)
    return ((risk_contribution(w, cov) - target_risk) ** 2).sum()

    
def risk_optimization(cov):
    """
    Provided a covariance matrix, gives you the weights of the portfolio
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    
    # constraints: unitarity of the weights and positivity
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w) -1}, {'type':'ineq','fun': lambda w: w})

    w = minimize(optim_func, init_guess, args=(cov), method='SLSQP', constraints=cons)
    return w.x


def get_data(assets,start, end):
    # Downloading data
    data = yf.download(assets, start=start, end=end)
    data = data.loc[:, ('Adj Close', slice(None))]
    return data
    

def get_returns(start, end, prevstart, prevend):
    # Ticker of the assets: GLD (Gold), TLT (Long Term Bonds), SPY (SP500)
    assets = ['GLD','TLT','SPY']
    assets.sort()

    # Downloading data
    data = get_data(assets,start,end) # download data for this year
    prevdata = get_data(assets,prevstart, prevend) # download data of previous year
    prevdata.columns = assets
    data.columns = assets

    # get returns and covariance matrix on previous year to not have lookahead bias
    prevreturns = prevdata[assets].pct_change().dropna()
    cov = prevreturns.cov()

    # get returns of new year
    returns = data[assets].pct_change().dropna()

    # compute the weights and calculate the returns for the year
    weights = risk_optimization(cov.values)
    returns['portfolio'] = sum(returns[assets[i]]*weights[i] for i in range(len(assets)))

    return returns


def annualised_sharpe(returns, N=252):
    return np.sqrt(N)* returns.mean() / returns.std()


if __name__ == '__main__':


    total_ret = []
    

    
    for i in range(2005, 2021):
        prevstart = str(i-1)+'-01-01'
        prevend   = str(i-1)+'-12-31'
        start = str(i)+'-01-01'
        end   = str(i)+'-12-31'

        # get the returns of the year
        ret = get_returns(start, end, prevstart, prevend)
        total_ret.append(ret)
        


    df_total = pd.concat(total_ret)

    print("Sharpe Ratio of the Portfolio:", annualised_sharpe(df_total.portfolio))
    print("Sharpe Ratio of the SP500:", annualised_sharpe(df_total.SPY))
    
    # plot the overall
    total_ret_portf = (1+df_total.portfolio).cumprod()
    total_ret_bench = (1+df_total.SPY).cumprod()


    amount = 100
    plt.plot(total_ret_portf*amount, label='risk parity')
    plt.plot(total_ret_bench*amount, label='SP500')
    plt.legend()

    plt.savefig('simple_risk_parity.png')
