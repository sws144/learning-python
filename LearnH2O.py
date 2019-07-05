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
h2o_df["CAPSULE"] = h2o_df["CAPSULE"].asfactor()
h2o_df.summary()

# choosing model
# L-BGFS for larger # of predictors
# IRLSM fewer predixtors


# stopping criteria
h2o_df = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/allyears2k_headers.zip")

# stops the model when we reach 10 predictors 
model = H2OGeneralizedLinearEstimator(family = "binomial", lambda_search = True, max_active_predictors = 10)
model.train(y = "IsDepDelayed", x = ["Year", "Origin"] , training_frame = h2o_df)
print(model)

# k-fold validation
h2o_df = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv")
# h2o.export_file(h2o_df, "test.csv")
h2o_df['CAPSULE'] = h2o_df['CAPSULE'].asfactor()
binomial_fit = H2OGeneralizedLinearEstimator(family = "binomial" , nfolds = 5, fold_assignment = "Random")
binomial_fit.train(y = "CAPSULE", x = ["AGE", "RACE", "PSA", "GLEASON"], training_frame = h2o_df)
print ("training auc: ") , binomial_fit.auc(train=True)
print ("cross-validation auc: "), binomial_fit.auc(xval=True)

# grid search over alphas (to weight between lasso 1 & ridge 2)
h2o_df = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv")
h2o_df[’CAPSULE’] = h2o_df[’CAPSULE’].asfactor()
alpha_opts = [0.0, 0.25, 0.5, 1.0]
hyper_parameters = {"alpha": alpha_opts}

# import grid search
from h2o.grid.grid_search import H2OGridSearch

grid = H2OGridSearch(H2OGeneralizedLinearEstimator(family="binomial"),hyper_params = hyper_parameters)
grid.train(y="CAPSULE", x = ["AGE", "RACE", "PSA", "GLEASON"], training_frame = h2o_df)
for m in grid:
    print("Model ID:" + m.model_id + " auc: " , m.auc())
    print(m.summary())
    print("\n\n")

# grid search over lambda (for regularization)
h2o_df = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv")
h2o_df['CAPSULE'] = h2o_df['CAPSULE'].asfactor()
lambda_opts = [1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0]
hyper_parameters = {"lambda:": lambda_opts}

grid = H2OGridSearch(H2OGeneralizedLinearEstimator(family="binomial"), hyper_params = hyper_parameters)
grid.train(y = "CAPSULE", x = ["AGE", "RACE", "PSA", "GLEASON"], training_frame = h2o_df)
for m in grid:
    print("Model ID:" + m.model_id + " auc: " , m.auc())
    print(m.summary())
    print("\n\n")

# GLM output logistic
h2o_df = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv")
h2o_df['CAPSULE'] = h2o_df['CAPSULE'].asfactor()

#make train & validation sets
r = h2o_df[0].runif(seed=1234)
train = h2o_df[r <= 0.8]
valid = h2o_df[r > 0.8]
binomial_fit = H2OGeneralizedLinearEstimator(family = 'binomial')
binomial_fit.train(y = "CAPSULE", x = ["AGE", "RACE","PSA", "GLEASON"], training_frame = train, validation_frame = valid)
print(binomial_fit)


# coefficients
binomial_fit.pprint_coef()
sorted(binomial_fit.coef_norm().items(), key=lambda x:x[1], reverse=True)

# model statistics
binomial_fit.summary()
binomial_fit._model_json["output"]["model_summary"].__getitem__('number_of_iterations')
binomial_fit.null_degrees_of_freedom(train=True, valid=True)
binomial_fit.residual_degrees_of_freedom(train=True, valid=True)
binomial_fit.mse(train=True, valid=True)
binomial_fit.r2(train=True, valid=True)
binomial_fit.logloss(train=True, valid=True)
binomial_fit.auc(train=True, valid=True)
# binomial_fit.giniCoef(train=True, valid=True)  #doesn't work
binomial_fit.null_deviance(train=True, valid=True)
binomial_fit.aic(train=True, valid=True)

# confusion matrix
binomial_fit.confusion_matrix(valid = False)
binomial_fit.confusion_matrix(valid = True)

# scoring
binomial_fit.scoring_history

# making predictions
h2o_df = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv")
h2o_df['CAPSULE'] = h2o_df['CAPSULE'].asfactor()

rand_vec = h2o_df.runif(1234)



