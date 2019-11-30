## This script evaluates the overtime performance if mean of the dependent variable berta_0 changes

import sys

path = '/Users/oksanahrytsiv/Desktop/Thesis/statistical-VS-ml-methods-performance/src' #change to your path here
sys.path.append(path)


# Import scripts / objects
from main import my_models, parameters
from train_init_dataset import df, samples_list, scores_mlr, scores_rfr, scores_gbr
from SimulationModel import SimulationModel
from Evaluation import Evaluation

init_timestamp = 2019

# Define data generation parameters for year 2019
N = 1000 # Observations in the population
n = 100  # Observations in each sample
k = 10   # Number of samples
t = 4    # Number of years

parameters = [N, n, k, t]


# Define regressions coefficients for years 2 to 5
beta0_y2                     = 0.2
beta1_y2, beta2_y2, beta3_y2 = 1, 1, 1
error_sd_y2                  = 1
coefficients_y2 = [beta0_y2, beta1_y2, beta2_y2, beta3_y2, error_sd_y2]
#-------------------------------
beta0_y3                     = 0.4
beta1_y3, beta2_y3, beta3_y3 = 1, 1, 1
error_sd_y3                  = 1
coefficients_y3 = [beta0_y3, beta1_y3, beta2_y3, beta3_y3, error_sd_y3]
#-------------------------------
beta0_y4                     = 0.6
beta1_y4, beta2_y4, beta3_y4 = 1, 1, 1
error_sd_y4                  = 1
coefficients_y4 = [beta0_y4, beta1_y4, beta2_y4, beta3_y4, error_sd_y4]
#-------------------------------
beta0_y5                     = 0.8
beta1_y5, beta2_y5, beta3_y5 = 1, 1, 1
error_sd_y5                  = 1
coefficients_y5 = [beta0_y5, beta1_y5, beta2_y5, beta3_y5, error_sd_y5]

#-------------------------------------------------------------------------------

# Initialize a list of coefficients for each year
coefficients = [coefficients_y2, coefficients_y3, coefficients_y4, coefficients_y5]


# Initialize a dictionaries of accuracy metrics for year 1
population_scores_mlr = {init_timestamp : scores_mlr}
population_scores_rfr = {init_timestamp : scores_rfr}
population_scores_gbr = {init_timestamp : scores_gbr}

# Initialize a collection of year1 population as a dictionaty
populations_collection = {init_timestamp : df}

# Initialize a collection of samples from population of year1
samples_list_collection = {init_timestamp : samples_list}

simulation_obj = SimulationModel()

populations_collection = simulation_obj.simulate_next_populations('study3', init_timestamp, parameters, coefficients, populations_collection)

samples_list_collection = simulation_obj.create_samples_collection(init_timestamp, parameters, populations_collection, samples_list_collection)

eval_obj = Evaluation()

population_scores_mlr, population_scores_rfr, population_scores_gbr = eval_obj.train(init_timestamp, parameters, population_scores_mlr, population_scores_rfr, population_scores_gbr, samples_list_collection)

# Now we create plots that visualize MSE of each model for a timespan of t years
eval_obj.create_plot_MSE(init_timestamp, population_scores_mlr, population_scores_rfr, population_scores_gbr, 'Study 3: MSE overtime')
