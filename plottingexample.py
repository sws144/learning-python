# plotting matplot lib
# http://actuarialdatascience.com/matplotlib-nice-plot.html

# %% packages
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

#%% check settings and style 
matplotlib.get_configdir()
matplotlib.style.use('classic') #to make sure is white
# matplotlib.style.use('dark_background') #backup/default is dark

# %% read data 
# assumes file is in \\learning-python
xlswb  = 'learning-python\\data_zero_bond_yield_curves.xlsx' 
data = pd.read_excel(xlswb, index_col='year')

#%% create plot
fig, ax  = plt.subplots()
fig.set_dpi(720)

ax.plot(data.index, data['Zero jan 2018'])
ax.plot(data.index, data['Zero dec 2018'])
ax.set_title('In 2018 pension funds suffered from decreasing rates')
ax.set_xlabel('maturity (years)')
ax.set_ylabel('rate (%)')
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=-0.5, ymax=2.75)
ax.annotate(s='January', xy=(82, 2.35), color='tab:blue', size=8)
ax.annotate(s='December', xy=(82, 1.8), color='tab:orange', size=8)

#%% save plot

fig.savefig('learning-python/plottingexample.png') #backslash only needs one, can use \\ as well 