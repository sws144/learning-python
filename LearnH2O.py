# Learning H2O
# https://h2o-release.s3.amazonaws.com/h2o/rel-ueno/2/docs-website/h2o-docs/booklets/GLMBooklet.pdf

# code to auto add packages
import subprocess
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# install packages
install("H2O")

import h2o

# Start H2O on local machine
h2o.init()

# Get help
help(h2o.estimators.glm.H2OGeneralizedLinearEstimator)

