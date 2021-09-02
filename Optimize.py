import numpy as np
import pandas as pd
import quandl
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

"""
Note this program uses the following Kaggle Dataset:
https://www.kaggle.com/jacksoncrow/stock-market-dataset
Here etfs folder has data about many different ETFs and stokcs folder has data about many different stokcs. 
"""


# Ask the User for the stock and ETF symbols
#---------------------------------------------------
def get_stock_list():
  print("Please enter the stock/ETF symbols you want in your portfolio (Ex: AMZN)")
    while True:
        print("Please Press q to exit")
        stock=  input("Enter your stock")
        if stock == "q":
          break
        stock_list.append(stock)
        
get_stock_list()      
#stock_list = ["DIS", "KO", "PEP", "WMT", "IVV", "VUG"]

# Get The Data
#---------------
data_list = []

for stock in stock_list:
  # Following if-else enables you to enter stock and ETF symbols at the same time. 
  if os.path.isfile('etfs/' + str(stock).upper() + ".csv"):
      data = pd.read_csv('etfs/' + stock.upper() + '.csv',  index_col='Date',parse_dates=True)
  else:
      data = pd.read_csv("stocks/" + stock.upper() + ".csv",  index_col='Date',parse_dates=True)
  data = pd.DataFrame(data["Close"]) # We only want the closing values
  data = data["2012-01-01":"2017-01-02"] # Date can be adjusted here in the format: Year-Month-Day
  data_list.append(data)

  
# Create the Data Frame
#---------------------------
stocks = pd.concat(data_list, axis=1)
stocks.columns = [name for name in stock_list]
stocks.head()

# Get Logarithmic Returns
log_ret = np.log(stocks/stocks.shift(1))
log_ret.head()


# Create The Necessary Functions
#-----------------------------------

def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * -1
# Contraints
def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1
def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio
    """
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])



# Set up the bounds & Optimize
#-----------------------------------

# 0-1 bounds for each weight needs to be in the form of a tuple. 
bounds = [(0,1) for i in range(len(stock_list))]
bounds = tuple(bounds)

# Initial Guess (equal distribution)
eql = 1.0/len(stock_list)
init_guess = [eql for i in range(len(stock_list))]
cons = ({'type':'eq','fun': check_sum})

# Sequential Least SQuares Programming (SLSQP).
opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)

print("Your Optimum Wegihts are:")
for stock, weight in zip(stock_list, opt_results.x):
    print(stock + ": " + str(weight))
print("--------------------------------------")
print("Here are the details of your optimum results:")
print(opt_results)



