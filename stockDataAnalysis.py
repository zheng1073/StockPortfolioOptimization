#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import pandas_datareader as dr
import matplotlib.pyplot as plt
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
import datetime

start = datetime.datetime(2019,8,1)
end = datetime.datetime(2020,1,1)

stock_data = dr.data.get_data_yahoo(['AAPL', 'F', 'IBM', 'AMZN'], start, end) #(stock name, start, end)
selected = stock_data["Adj Close"]
print(selected)

selected.plot()
plt.xlabel('Quarter')
plt.ylabel('Value ($)')
plt.show()

returns_quarterly = df.pct_change()
expected_returns = returns_quarterly.mean()
cov_quarterly = returns_quarterly.cov()
print(cov_quarterly)

def return_portfolios(expected_returns, cov_matrix):
    port_returns = []
    port_volatility = []
    stock_weights = []
    
    selected = (expected_returns.axes)[0]
    
    num_assets = len(selected) 
    num_portfolios = 500
    
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)
    
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility}
    
    for counter,symbol in enumerate(selected):
        portfolio[symbol +' Weight'] = [Weight[counter] for Weight in stock_weights]
    
    df = pd.DataFrame(portfolio)
    
    column_order = ['Returns', 'Volatility'] + [stock+' Weight' for stock in selected]
    
    df = df[column_order]
   
    return df

random_portfolios = return_portfolios(expected_returns, cov_quarterly)
print(random_portfolios)

random_portfolios.plot.scatter(x='Volatility', y='Returns', figsize = (10,5))
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()

