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
b = 3   # Number of predictor variables (min = 3)


beta_change_study1 = 1.4 # Strength of beta coefficient increase
relationship_term_study2 = 0.1
target_mean_study3 = 10

defaults = {'n_rows': N, 'n_rows_sample': n, 'n_samples': k, 'n_years': t, \
 'start_year': y, 'n_X': b}

# Identify linear or non-linear function based on user input
complexity = input("Choose which regression function to test (linear or polynomial): ")

# Calling object GeneratePopulation() will create a dataset for year 2019 and save it in a folder 'data'
obj = GeneratePopulation(defaults)

# Get the coefficients from a generated population
coefficients_y1 = obj.generate_population(complexity)

# Import dataset
dataset = pd.read_csv('data/init_population_' + complexity + '.csv', sep = ",")

# Initialize a dictionary where key is a year and value is a dictionary of coefficients
coefficients = {defaults['start_year']: coefficients_y1}


# # Compute mean of the population
# print(simple_dataset.mean())

study = input("Choose which study to run (study1, study2, study3 or study4): ")

# Run study based on user input
if study == 'study1':
    study1 = Study_1(defaults, coefficients, dataset, beta_change_study1, complexity)
    study1.run_simulation()
elif study == 'study2':
    study2 = Study_2(defaults, coefficients, dataset, relationship_term_study2, complexity)
    study2.run_simulation()
elif study == 'study3':
    study3 = Study_3(defaults, coefficients, dataset, target_mean_study3, complexity)
    study3.run_simulation()
elif study == 'study4':
    study4 = Study_4(defaults, coefficients, dataset, beta_change_study1, \
                     relationship_term_study2, target_mean_study3, complexity)
    study4.run_simulation()
else: 
    print('This study does not exist. Please select an existing study')    
    
