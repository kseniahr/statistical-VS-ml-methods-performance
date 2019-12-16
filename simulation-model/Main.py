#import Class of each study
#-------------------------------------------------------------------------------
from Study_1 import Study1
from Study_2 import Study2
from Study_3 import Study3
from Study_4 import Study4
from CreatePopulation import CreatePopulation

# Import libraries
import pandas as pd

# Define data generation parameters for t years
N = 1000 # Observations in the population
n = 100  # Observations in each sample
k = 10   # Number of samples
t = 5    # Number of years
y = 2019 # Starting year

parameters = [N, n, k, t, y]

# Calling object CreatePopulation() will create a dataset for year 2019 and save it in a folder 'data'
CreatePopulation(parameters)

# Import dataset A
df = pd.read_csv('data/init_dataset.csv', sep = ",")

# Import complex dataset B

# Run each simulation study
Study1(parameters, df)
Study2(parameters, df)
Study3(parameters, df)
Study4(parameters, df)
