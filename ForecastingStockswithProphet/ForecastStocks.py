# Prophet for stock forecasting

#%% packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# REQUIRES Microsoft Visual C++ Build Tools for Prophet
from fbprophet import Prophet

import statsmodels.api as sm
from scipy import stats
from pandas.core import datetools

from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings("ignore")

# plt.style.available
plt.style.use("seaborn-whitegrid")

# todo....