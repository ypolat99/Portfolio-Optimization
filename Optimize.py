import numpy as np
import pandas as pd
import quandl
import matplotlib.pyplot as plt
%matplotlib inline


# Download and Get Daily Returns
start = pd.to_datetime('2012-01-01')
end = pd.to_datetime('2017-01-01')
