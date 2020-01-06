from GeneratePopulation import GeneratePopulation
from Study_1 import Study_1
from Study_2 import Study_2
from Study_3 import Study_3
from Study_4 import Study_4


# Import libraries
import time
import pandas as pd

# Define data generation settings:

n_X           = 3           # Number of predictor variables (min = 3)
start_year    = 2019        # Starting year
n_years       = 10          # Number of years

beta_change_study1 = 1.6 # Strength of beta coefficient increase
corr_term_study2   = 0.1
target_mean_study3 = 10 

user_input1   = input("Select type of a regression function: linear or polynomial => ")
user_input2   = input("Choose type of a variable: continuous or hybrid => ")
n_rows        = int(input("Number of observations: => "))
n_samples     = int(input("Number of samples: => "))
n_rows_sample = int(input("Number of observations per sample: => "))


if n_X <= 3:
    dimensionality = 'L' # low dimensionality of a dataset
elif n_X > 3:
    dimensionality = 'H' # high dimensionality of a dataset
else:
    raise Exception('Number of dimensions should not be negative. The value of n_X was: {}'.format(n_X)) # error message


if user_input1 == 'linear':
    complexity = 'L'
elif user_input1 == 'polynomial':
    complexity = 'P'
else: 
    raise Exception('This type of complexity does not exist') # error message


if user_input2 == 'continuous':
    var_type = 'C'
elif user_input2 == 'hybrid':
    var_type = 'H'
else: 
    raise Exception('This type of variables is incorrect. The value of var_type was: {}'.format(user_input2)) # error message
    
defaults = {'n_rows': n_rows, 'n_rows_sample': n_rows_sample, 'n_samples': n_samples, \
            'n_years': n_years, 'start_year': start_year, 'n_X': n_X}

start = time.time() 

obj = GeneratePopulation(defaults)

# Get the coefficients from a generated population
coefficients_y1 = obj.generate_population(dimensionality, complexity, var_type)

print(coefficients_y1)

generation_time = time.time()
print('Data Generation Time: % 2d s' %(generation_time - start)) # Time in seconds

# Import dataset
dataset = pd.read_csv('data/' + dimensionality + complexity + var_type + '.csv', sep = ",")

print(dataset.describe(include = 'all'))

# Initialize a dictionary where key is a year and value is a dictionary of coefficients
coefficients = {defaults['start_year']: coefficients_y1}

study = input("Choose which study to run (study1, study2, study3 or study4): ")

# Track execution time
start = time.time()

# Run study based on user input
if study == 'study1':
    study1 = Study_1(defaults, coefficients, dataset, beta_change_study1, dimensionality, complexity, var_type)
    study1.run_simulation()
elif study == 'study2':
    study2 = Study_2(defaults, coefficients, dataset, corr_term_study2, dimensionality, complexity, var_type)
    study2.run_simulation()
elif study == 'study3':
    study3 = Study_3(defaults, coefficients, dataset, target_mean_study3, dimensionality, complexity, var_type)
    study3.run_simulation()
elif study == 'study4':
    study4 = Study_4(defaults, coefficients, dataset, beta_change_study1, \
                     corr_term_study2, target_mean_study3, dimensionality, complexity, var_type)
    study4.run_simulation()
else:
    print('This study does not exist. Please select an existing study')
    
end = time.time() 

print('Execution Time: % 2d s' %(end - start)) # Time in seconds


