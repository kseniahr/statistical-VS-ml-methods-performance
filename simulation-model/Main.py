#import Class of each study
#-------------------------------------------------------------------------------
from Study_1 import Study1
from Study_2 import Study2
from Study_3 import Study3
from Study_4 import Study4
from CreatePopulation import CreatePopulation
from GeneratePopulation import GeneratePopulation

# Import libraries
import pandas as pd

# Define data generation defaults for t years
N = 1000 # Observations in the population
n = 100  # Observations in each sample
k = 10   # Number of samples
t = 5    # Number of years
y = 2019 # Starting year
b = 10   # Number of predictor variables

defaults = {'n_rows': N, 'n_rows_sample': n, 'n_samples': k, 'n_years': t, 'start_year': y, 'num_X': b}

# Calling object GeneratePopulation() will create a dataset for year 2019 and save it in a folder 'data'
#GeneratePopulation(defaults)

# Calling object CreatePopulation() will create a dataset for year 2019 and save it in a folder 'data'
CreatePopulation(defaults)

# Import dataset A
#simple_dataset = pd.read_csv('data/init_population_A.csv', sep = ",")
simple_dataset = pd.read_csv('data/init_dataset.csv', sep = ",")
# Import complex dataset B
#complex_dataset = pd.read_csv('data/init_population_B.csv', sep = ",")

# Run each simulation study
Study1(defaults, simple_dataset)
Study2(defaults, simple_dataset)
Study3(defaults, simple_dataset)
Study4(defaults, simple_dataset)
