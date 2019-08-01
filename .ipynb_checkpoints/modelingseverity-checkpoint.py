# modeling severity
# https://statcompute.wordpress.com/2015/12/06/modeling-severity-in-operational-losses-with-python/

# code to auto add packages
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# install packages
install("pandas")
install("numpy")
install("statsmodels")

# import
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("Autocollision.csv")  # need to get larger dataset
df.head()

# fit a gamma regression
gamma = smf.glm(formula = "Severity ~ Age + Vehicle_Use", data = df, 
    family = sm.families.Gamma(sm.families.links.log) )
type(gamma)
gamma.fit().summary()