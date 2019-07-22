# Cohort analysis with python
 
# code to auto add packages
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

install("pandas")
install("numpy")
install("matplotlib")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

pd.set_option('max_columns', 50)
mpl.rcParams['lines.linewidth'] = 2

#%matplotlib inline

# import data
df = pd.read_excel('chapter-12-relay-foods.xlsx', sheet_name="Pilot Study Data")
df.head()

# 1 create period column based on order date
df['OrderPeriod'] = df.OrderDate.apply(lambda x : x.strftime('%Y-%m'))
df.head()

# 2 determine user's cohort group
df.set_index('UserId', inplace = True)
type(df)
df['CohortGroup'] = df.groupby(level=0)['OrderDate'].min().apply(lambda x : x.strftime('%Y-%m'))
df.reset_index(inplace=True)
df.head()

# 3 Rollup data by CohortGroup & OrderPeriod
