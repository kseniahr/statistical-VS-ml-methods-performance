## This script evaluates the overtime performance of all 3 studies combined


# Import libaries
import pandas as pd
import numpy as np
import random

# Import scripts with all functions
import functions as f
from main import independent_vars, my_models, parameters
from train_init_dataset import df, samples_list, scores_mlr, scores_rfr, scores_gbr

#-------------------------------------------------------------------------------

# Define regressions coefficients for the 1st year
intercept_y1                 = 0
beta1_y1, beta2_y1, beta3_y1 = 1, 1, 1
error_sd_y1                  = 1

# Create a list of predefined coefficients
coefficients_y1 = [intercept_y1, beta1_y1, beta2_y1, beta3_y1, error_sd_y1]

#-------------------------------------------------------------------------------

# Define regressions coefficients for years 2 to 5
intercept_y2                 = 0.2
beta1_y2, beta2_y2, beta3_y2 = 0.9, 1.2, 1.1
error_sd_y2                  = 1
coefficients_y2 = [intercept_y2, beta1_y2, beta2_y2, beta3_y2, error_sd_y2]
#-------------------------------
intercept_y3                 = 0.4
beta1_y3, beta2_y3, beta3_y3 = 0.7, 1.4, 1.3
error_sd_y3                  = 1
coefficients_y3 = [intercept_y3, beta1_y3, beta2_y3, beta3_y3, error_sd_y3]
#-------------------------------
intercept_y4                 = 0.6
beta1_y4, beta2_y4, beta3_y4 = 0.5, 1.6, 1.5
error_sd_y4                  = 1
coefficients_y4 = [intercept_y4, beta1_y4, beta2_y4, beta3_y4, error_sd_y4]
#-------------------------------
intercept_y5                 = 0.8
beta1_y5, beta2_y5, beta3_y5 = 0.3, 1.8, 1.7
error_sd_y5                  = 1
coefficients_y5 = [intercept_y5, beta1_y5, beta2_y5, beta3_y5, error_sd_y5]

#-------------------------------------------------------------------------------

# Initialize a list of coefficients for each year
coefficients = [coefficients_y2, coefficients_y3, coefficients_y4, coefficients_y5]

# Initialize a dictionaries of accuracy metrics for year 1
population_scores_mlr = {'year1' : scores_mlr}
population_scores_rfr = {'year1' : scores_rfr}
population_scores_gbr = {'year1' : scores_gbr}

# Initialize a collection of year1 population as a dictionaty
populations_collection = {'year1' : df}

# Initialize a collection of samples from population of year1
samples_list_collection = {'year1' : samples_list}

# This loop simulates future populations for t years. Also it creates k samples for each year with n observations in each sample.
for t in range (0, parameters[3]):

    year_key = 'year' + str(t+2)
    prev_year_key = 'year' + str(t+1)

    # Generate exogene variables (including latent variables like prediction-error)
    independent_vars, error, Y = f.generate_vars(parameters, coefficients[t], populations_collection[prev_year_key], year_key)

    # Combine dependent and independent variables in a data-frame
    populations_collection[year_key] = pd.DataFrame({'X1': independent_vars[0], 'X2': independent_vars[1], 'X3': independent_vars[2], 'Y': Y, 'error': error})

    # Create k samples with n observations using random.sample
    samples_list_collection[year_key] = f.draw_k_samples(parameters, populations_collection[year_key])

# Next we want to fit chosen models on each sample of t populations
# Linear Regression, Random Forest Regressor, Gradient Boosting Regressor

# Fit all models to the samples of next t=4 years
for t in range (0, parameters[3]):
    year_key = 'year' + str(t+2)
    population_scores_mlr[year_key] =  f.fit_lr(parameters, samples_list_collection[year_key], my_models[0])
    population_scores_rfr[year_key] =  f.fit_rfr(parameters, samples_list_collection[year_key], my_models[1])
    population_scores_gbr[year_key] =  f.fit_gbr(parameters, samples_list_collection[year_key], my_models[2])

# Now we create histograms that visualize the distribution of feature X1 overtime:
f.create_histograms(populations_collection, 'Study 4: concept drift')

# Now we create plots that visualize MSE of each model for a timespan of t years
f.create_plot_MSE(population_scores_mlr, population_scores_rfr, population_scores_gbr, 'Study 4: MSE with concept drift overtime')
