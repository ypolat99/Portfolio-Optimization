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

# Get The Data
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

  
#Create the Data Frame
stocks = pd.concat(data_list, axis=1)
stocks.columns = [name for name in stock_list]
stocks.head()


