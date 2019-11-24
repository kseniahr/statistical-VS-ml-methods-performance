# Import Machine Learning models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Initialize X1, X2 and X3 with zeros as a starting point
independent_vars = [0,0,0]

#-------------------------------------------------------------------------------

# Define which models are going to be tested
mlr = LinearRegression() # statistical model
rfr  = RandomForestRegressor() # ML model
gbr  = GradientBoostingRegressor() # ML model

my_models = [mlr, rfr, gbr]

#-------------------------------------------------------------------------------

# Define data generation parameters for year 1
N = 1000 # Observations in the population
n = 100  # Observations in each sample
k = 10   # Number of samples
t = 4    # Number of years

parameters = [N, n, k, t]

import study_1
import study_2
import study_3

import study_combined
