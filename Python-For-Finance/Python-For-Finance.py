# https://towardsdatascience.com/python-for-finance-stock-portfolio-analyses-6da4c3e61054

# %% 1 Import initial libraries
import pandas as pd
import xlrd #for excel import

import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Imports in order to be able to use Plotly offline.
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


print(__version__)  # requires version >= 1.9.0

#init_notebook_mode(connected=True) # for jupyter notebook

# Import the Sample worksheet with acquisition dates and initial cost basis:

# %% 2 read in data
portfolio_df = pd.read_excel('Sample_stocks_acquisition_dates_costs.xlsx', 
                             sheet_name='Sample')
portfolio_df.head(10)


# Date Ranges for SP 500 and for all tickers
# Modify these date ranges each week.

# The below will pull back stock prices from the start date until end date specified.
start_sp = datetime.datetime(2013, 1, 1)
end_sp = datetime.datetime(2018, 3, 9)

# This variable is used for YTD performance.
end_of_last_year = datetime.datetime(2017, 12, 29)

# These are separate if for some reason want different date range than SP.
stocks_start = datetime.datetime(2013, 1, 1)
stocks_end = datetime.datetime(2018, 3, 9)

# Leveraged from the helpful Datacamp Python Finance trading blog post.
from pandas_datareader import data as pdr
sp500 = pdr.get_data_yahoo('^GSPC', 
                           start_sp,
                             end_sp)
    
sp500.head()

# %% 3 Generate a dynamic list of tickers
# to pull from Yahoo Finance API based on the imported file with tickers.
