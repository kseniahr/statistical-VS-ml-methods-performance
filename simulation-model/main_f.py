# Import libaries
import pandas as pd
import numpy as np
import random
# Import scripts
import test as f
# Import Machine Learning models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Call np.random.seed() to make examples with (pseudo) random data reproducible:



# Define data generation parameters for year 1
N = 10000 # Observations in the population
n = 1000  # Observations in each sample
k = 10   # Number of samples
t = 5    # Number of years

parameters = [N, n, k, t]

# Initialize X1, X2 and X3 with zeros as a starting point
independent_vars = [0,0,0]

#-------------------------------------------------------------------------------

# Define regressions coefficients for the 1st year
intercept_y1                 = 0
beta1_y1, beta2_y1, beta3_y1 = 1, 0, -1
error_sd_y1                  = 1

# Create a list of predefined coefficients
coefficients_y1 = [intercept_y1, beta1_y1, beta2_y1, beta3_y1, error_sd_y1]

#-------------------------------------------------------------------------------

# Define regressions coefficients for years 2 to 5
intercept_y2                 = 0
beta1_y2, beta2_y2, beta3_y2 = 1.6, 0, -1
error_sd_y2                  = 1
coefficients_y2 = [intercept_y2, beta1_y2, beta2_y2, beta3_y2, error_sd_y2]
#-------------------------------
intercept_y3                 = 0
beta1_y3, beta2_y3, beta3_y3 = 2.56, 0, -1
error_sd_y3                  = 1
coefficients_y3 = [intercept_y3, beta1_y3, beta2_y3, beta3_y3, error_sd_y3]
#-------------------------------
intercept_y4                 = 0
beta1_y4, beta2_y4, beta3_y4 = 4, 0, -1
error_sd_y4                  = 1
coefficients_y4 = [intercept_y4, beta1_y4, beta2_y4, beta3_y4, error_sd_y4]
#-------------------------------
intercept_y5                 = 0
beta1_y5, beta2_y5, beta3_y5 = 6.55, 0, -1
error_sd_y5                  = 1
coefficients_y5 = [intercept_y5, beta1_y5, beta2_y5, beta3_y5, error_sd_y5]

#-------------------------------------------------------------------------------

# Initialize a list of coefficients for each year
coefficients = [coefficients_y1, coefficients_y2, coefficients_y3, coefficients_y4, coefficients_y5]

# Initialize a collection of all t populations as a dictionaty
populations_collection = {}

# Initialize a collection of samples from each population
samples_list_collection = {}


# This loop simulates future populations for t years. Also it creates k samples for each year with n observations in each sample.
for t in range (0, parameters[3]):

    # Generate exogene variables (including latent variables like prediction-error)
    independent_vars, error, Y = f.generate_vars(parameters, coefficients[t], independent_vars)

    # Combine dependent and independent variables in a data-frame
    populations_collection[t] = pd.DataFrame({'X1': independent_vars[0], 'X2': independent_vars[1], 'X3': independent_vars[2], 'Y': Y, 'error': error})

    # Create k samples with n observations using random.sample
    samples_list_collection[t] = f.draw_k_samples(parameters, populations_collection[t])


# Next we want to fit chosen models on each sample of t populations
# Linear Regression, Random Forest Regressor, Gradient Boosting Regressor


mlr = LinearRegression()
rfr  = RandomForestRegressor()
gbr  = GradientBoostingRegressor()

my_models = [mlr, rfr, gbr]

population_scores_mlr = {}
population_scores_rfr = {}
population_scores_gbr = {}

#---------------------------------------------------------
# Fit Linear Regression to all samples of Y1
#---------------------------------------------------------
mlr_model = {}

for t in range (0, parameters[3]):
    population_scores_mlr[t], mlr_model =  f.fit_lr(parameters, samples_list_collection[t], my_models[0], t, mlr_model)
    population_scores_rfr[t] =  f.fit_rfr(parameters, samples_list_collection[t], my_models[1])
    population_scores_gbr[t] =  f.fit_gbr(parameters, samples_list_collection[t], my_models[2])


print(list(population_scores_mlr.values()))
print(list(population_scores_rfr.values()))
print(list(population_scores_gbr.values()))


import numpy as np
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)

pl.figure()
ax = pl.subplot(gs[0, 0]) # row 0, col 0
pl.plot([1,2,3,4,5],[population_scores_gbr[0][0],population_scores_gbr[1][0],population_scores_gbr[2][0],population_scores_gbr[3][0], population_scores_gbr[4][0]])
pl.title('GradientBoostingRegressor', fontsize = 8)
pl.xlabel('years')
pl.ylim(0, 3)

ax = pl.subplot(gs[0, 1]) # row 0, col 1
pl.plot([1,2,3,4,5],[population_scores_rfr[0][0],population_scores_rfr[1][0],population_scores_rfr[2][0],population_scores_rfr[3][0], population_scores_rfr[4][0]])
pl.title('RandomForestRegressor', fontsize = 8)
pl.xlabel('years')
pl.ylim(0, 3)

ax = pl.subplot(gs[1, :]) # row 1, span all columns
pl.plot([1,2,3,4,5],[population_scores_mlr[0][0],population_scores_mlr[1][0],population_scores_mlr[2][0],population_scores_mlr[3][0], population_scores_mlr[4][0]])
pl.title('LinearRegression', fontsize = 8)
pl.xlabel('years')
pl.ylim(0, 3)

pl.show()


show()
