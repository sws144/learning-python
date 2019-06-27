# logistic regression with apache spark
# https://medium.com/fuzz/understanding-logistic-regression-w-apache-spark-python-c32eae4d614e

# code to auto add packages
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# install packages
install("numpy")
install("pyspark")

import numpy as np 
from numpy import array 
from pyspark.mllib.regression import LabeledPoint 
