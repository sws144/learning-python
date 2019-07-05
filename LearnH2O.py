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
# help(h2o.estimators.glm.H2OGeneralizedLinearEstimator)
# help(h2o.estimators.gbm.H2OGradientBoostingEstimator)
# help(h2o.estimators.deeplearning.H2ODeepLearningEstimator)

# Show a demo
# h2o.demo("glm")
# h2o.demo("gbm")
# h2o.demo("deeplearning")

h2o.init(ip = "123.45.67.89", port = 54321)

# linear regression 
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
h2o.init()

h2o_df = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv")
gaussian_fit = H2OGeneralizedLinearEstimator(family = "gaussian")
gaussian_fit.train(y = "VOL", x = ["AGE", "RACE", "PSA", "GLEASON"],training_frame = h2o_df)

# logistic regression
binomial_fit = H2OGeneralizedLinearEstimator(family = "binomial")
binomial_fit.train(y = "CAPSULE", x = ["AGE", "RACE", "PSA", "GLEASON"], training_frame = h2o_df)

# multinomial  (by column number)
h2o_df = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris.csv")
multinomial_fit = H2OGeneralizedLinearEstimator(family = "multinomial")
multinomial_fit.train(y = 4, x = [0,1,2,3], training_frame = h2o_df)

# Poisson models
# use swedish insurance data
h2o_df = h2o.import_file(
    "http://h2o-public-test-data.s3.amazonaws.com/smalldata/glm_test/Motor_insurance_sweden.txt", sep = '\t')
poisson_fit = H2OGeneralizedLinearEstimator(family = "poisson")
poisson_fit.train(y="Claims", x= ["Payment", "Insured", "Kilometres", "Zone", "Bonus", "Make"], training_frame = h2o_df)
poisson_fit.coef()

# Gamma models
h2o_df = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv")
gamma_inverse = H2OGeneralizedLinearEstimator(family ="gamma", link = "inverse")
gammma_inverse.train(y = "DPROS", x = ["AGE", "RACE", "CAPSULE", "DCAPS", "PSA", "VOL"], training_frame = h2o_df)

gamma_log = H2OGeneralizedLinearEstimator(family = "gamma", link = "log")
gamma_log.train(y="DPROS", x= ["AGE", "RACE", "CAPSULE", "DCAPS", "PSA", "VOL"], training_frame = h2o_df)

# Tweedie 
# p= 0:  Normal
# p= 1:  Poisson 
# p∈(1,2):  Compound Poisson, non-negative with mass at zero
# p= 2:  Gamma
# p= 3:  Inverse-Gaussian
# p >2:  Stable, with support on the positive reals

h2o_df = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/glm_test/auto.csv")
tweedie_fit = H2OGeneralizedLinearEstimator(family = "tweedie")
tweedie_fit.train(y = "y", x = h2o_df.col_names[1:], training_frame=h2o_df)

# Building GLM models
h2o_df = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv")
