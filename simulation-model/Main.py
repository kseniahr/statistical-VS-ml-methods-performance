from GeneratePopulation import GeneratePopulation
from Study_1 import Study_1
# from Study_2 import Study2
# from Study_3 import Study3
# from Study_4 import Study4

# Import libraries
import pandas as pd

# Define data generation defaults for t:
N = 1000 # Observations in the population
n = 100  # Observations in each sample
k = 10   # Number of samples
t = 10    # Number of years
y = 2019 # Starting year
b = 10   # Number of predictor variables

beta_change = 1.2 # Strength of beta coefficient increase

defaults = {'n_rows': N, 'n_rows_sample': n, 'n_samples': k, 'n_years': t, 'start_year': y, 'n_X': b}

# Calling object GeneratePopulation() will create a dataset for year 2019 and save it in a folder 'data'
obj = GeneratePopulation(defaults)
coefficients_y1 = obj.generate_population()

# Import dataset A
simple_dataset = pd.read_csv('data/init_population_A.csv', sep = ",")
coefficients = {defaults['start_year']: coefficients_y1}

# Import complex dataset B
#complex_dataset = pd.read_csv('data/init_population_B.csv', sep = ",")
#
# Run each simulation study
study1 = Study_1(defaults, coefficients, simple_dataset, beta_change)
coefficients = study1.create_beta_coefs()
study1.run_simulation()
# #
# # Study2(defaults, simple_dataset)
# # Study3(defaults, simple_dataset)
# # Study4(defaults, simple_dataset)
