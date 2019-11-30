# Import Machine Learning models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


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
