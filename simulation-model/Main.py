from GeneratePopulation import GeneratePopulation
from Experiment_1 import Experiment_1
from Experiment_2 import Experiment_2
from Experiment_3 import Experiment_3
from Experiment_4 import Experiment_4


# Import libraries
import time
import pandas as pd

# Define data generation settings:

n_X           = 3           # Number of predictor variables (min = 3)
start_year    = 2019        # Starting year
n_years       = 10          # Number of years

beta_change_experiment1 = 1.3 # Strength of beta coefficient increase
corr_term_experiment2   = 0.1
target_mean_experiment3 = 2

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

experiment = int(input("Choose which experiment to run (1, 2, 3 or 4): "))

# Track execution time
start = time.time()

# Run experiment based on user input
if experiment   == 1:
    experiment1 = Experiment_1(defaults, coefficients, dataset, beta_change_experiment1, dimensionality, complexity, var_type)
    experiment1.run_simulation()
elif experiment == 2:
    experiment2 = Experiment_2(defaults, coefficients, dataset, corr_term_experiment2, dimensionality, complexity, var_type)
    experiment2.run_simulation()
elif experiment == 3:
    experiment3 = Experiment_3(defaults, coefficients, dataset, target_mean_experiment3, dimensionality, complexity, var_type)
    experiment3.run_simulation()
elif experiment == 4:
    experiment4 = Experiment_4(defaults, coefficients, dataset, beta_change_experiment1, \
                     corr_term_experiment2, target_mean_experiment3, dimensionality, complexity, var_type)
    experiment4.run_simulation()
else:
    print('This experiment does not exist. Please select an existing experiment')

end = time.time()

print('Execution Time: % 2d s' %(end - start)) # Time in seconds
