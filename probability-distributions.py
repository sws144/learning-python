# Probability Distributions 
# https://www.mikulskibartosz.name/monte-carlo-simulation-in-python/

# required packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

# Titanic dataset
dataset = seaborn.load_dataset('titanic')
RANDOM_STATE = 31415

ages = dataset.age.dropna()

# Uniform distribution
from scipy.stats import uniform
uniform_dist = uniform(loc = 0, scale = 20)
uniform_dist.rvs(size = 10, random_state = RANDOM_STATE)
x = np.linspace(-5, 25,100) 
_, ax = plt.subplots(1,1) #only use second argument so skip first
ax.plot(x,uniform_dist.pdf(x),'r--',lw=2)
plt.title('Uniform distribution of values between 0 and 20')
plt.show(block=False)   

