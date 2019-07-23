"""
Cohort analysis with python 
http://www.gregreda.com/2015/08/23/cohort-analysis-with-python/
"""

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
grouped = df.groupby(['CohortGroup', 'OrderPeriod'])

# count the unique users, orders, and total revenue per (group+period)
cohorts = grouped.agg({'UserId': pd.Series.nunique,
                        'Order Id': pd.Series.nunique,
                        'Total Charges': np.sum})

# make column more meaningful
cohorts.rename(columns={'UserId' : 'TotalUsers',
                        'Order Id' : 'TotalOrders'}, inplace = True)
cohorts.head()

# function for time
def cohort_period(df):
    """
    Creates a `CohortPeriod` column, which is the Nth period based on the user's first purchase.
    
    Example
    -------
    Say you want to get the 3rd month for every user:
        df.sort(['UserId', 'OrderTime', inplace=True)
        df = df.groupby('UserId').apply(cohort_period)
        df[df.CohortPeriod == 3]
    """
    df['CohortPeriod'] = np.arange(len(df)) + 1
    return df

cohorts = cohorts.groupby(level=0).apply(cohort_period)
cohorts.head()

# 5 Make sure we did that right
x = df[(df.CohortGroup == '2009-01') & (df.OrderPeriod == '2009-01')]
y = cohorts.ix[('2009-01', '2009-01')] #uses labels for new data

# shows nothing if true, these are checks
assert(x['UserId'].nunique() == y['TotalUsers']) 
assert(x['Total Charges'].sum().round(2) == y['Total Charges'].round(2))
assert(x['Order Id'].nunique() == y['TotalOrders'])

x = df[(df.CohortGroup == '2009-01') & (df.OrderPeriod == '2009-09')]
y = cohorts.ix[('2009-01', '2009-09')]

assert(x['UserId'].nunique() == y['TotalUsers'])
assert(x['Total Charges'].sum().round(2) == y['Total Charges'].round(2))
assert(x['Order Id'].nunique() == y['TotalOrders'])

x = df[(df.CohortGroup == '2009-05') & (df.OrderPeriod == '2009-09')]
y = cohorts.ix[('2009-05', '2009-09')]

assert(x['UserId'].nunique() == y['TotalUsers'])
assert(x['Total Charges'].sum().round(2) == y['Total Charges'].round(2))
assert(x['Order Id'].nunique() == y['TotalOrders'])

# user retention by Cohort Group

# reindex the DataFrame
cohorts.reset_index(inplace=True)
cohorts.set_index(['CohortGroup','CohortPeriod'], inplace=True)

cohort_group_size = cohorts['TotalUsers'].groupby(level=0).first()
cohort_group_size.head()

cohorts['TotalUsers'].head()

# unstack, like pivoting
cohorts['TotalUsers'].unstack(0).head()

# utilize broadcasting to divide each column by corresponding cohort_group_size
user_retention = cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)
user_retention.head(10)

# see plots
user_retention[['2009-02', '2009-03', '2009-04']].plot(figsize=(10,5))
plt.title('Cohorts: User Retention')
plt.xticks( np.arange(1,21.1,1))
plt.xlim(1,20)
plt.ylabel('% of cohort purchasing')
plt.show(block=False)

#for heatmaps
install("seaborn")
import seaborn as sns

plt.figure(figsize=(12,8))
plt.title('Cohort: User Retention')
sns.heatmap(user_retention.T, mask=user_retention.T.isnull(),annot=True, fmt='.0%') #.T is tranposed
plt.show(block=False)