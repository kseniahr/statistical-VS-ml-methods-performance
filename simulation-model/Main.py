from GeneratePopulation import GeneratePopulation
from Study_1 import Study_1
from Study_2 import Study_2
from Study_3 import Study_3
from Study_4 import Study_4


# Import libraries
import pandas as pd

# Define data generation defaults for t:
N = 1000 # Observations in the population
n = 100  # Observations in each sample
k = 10   # Number of samples
t = 10    # Number of years
y = 2019 # Starting year
b = 5   # Number of predictor variables (min = 3)

beta_change_study1 = 1.2 # Strength of beta coefficient increase
relationship_term_study2 = 0.1
target_mean_study3 = 0.8

defaults = {'n_rows': N, 'n_rows_sample': n, 'n_samples': k, 'n_years': t, \
 'start_year': y, 'n_X': b}

# # Calling object GeneratePopulation() will create a dataset for year 2019 and save it in a folder 'data'
obj = GeneratePopulation(defaults)
#
coefficients_y1 = obj.generate_population()

# Import dataset A
simple_dataset = pd.read_csv('data/init_population_A.csv', sep = ",")
coefficients = {defaults['start_year']: coefficients_y1}

# Import complex dataset B
#complex_dataset = pd.read_csv('data/init_population_B.csv', sep = ",")

# # Compute mean of the population
# print(simple_dataset.mean())

#Run each simulation study
study1 = Study_1(defaults, coefficients, simple_dataset, beta_change_study1)
study1.run_simulation()
#
# study2 = Study_2(defaults, coefficients, simple_dataset, relationship_term_study2)
# study2.run_simulation()
# study3 = Study_3(defaults, coefficients, simple_dataset, target_mean_study3)
# study3.run_simulation()
# study4 = Study_4(defaults, coefficients, simple_dataset, beta_change_study1, \
#  relationship_term_study2, target_mean_study3)
# study4.run_simulation()
