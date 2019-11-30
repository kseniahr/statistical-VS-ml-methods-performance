import sys

path = '/Users/oksanahrytsiv/Desktop/Thesis/statistical-VS-ml-methods-performance/src' #change to your path here
sys.path.append(path)

# Import helping functions
from functions import generate_variables

# Import libraries
import pandas as pd


# Define data generation parameters for year 1
N = 1000 # Observations in the population
n = 100  # Observations in each sample
k = 10   # Number of samples
t = 4    # Number of next populations (years)

parameters = [N, n, k, t]

#-------------------------------------------------------------------------------

# Define regressions coefficients for the 1st year
beta0_y1 =   0
beta1_y1 =   1
beta2_y1 =   1
beta3_y1 = (-1)
error_y1 =   1

# Create a list of predefined coefficients
coefficients_y1 = [beta0_y1, beta1_y1, beta2_y1, beta3_y1, error_y1]

#-------------------------------------------------------------------------------

# Generate exogene variables (including latent variables like prediction-error)
independent_vars, error, Y = generate_variables(parameters, coefficients_y1)

# Combine dependent and independent variables in a data-frame
df = pd.DataFrame({'X1': independent_vars[0], 'X2': independent_vars[1], 'X3': independent_vars[2], 'Y': Y, 'error': error})


# Export initial dataset for running the simulation
df.to_csv('dataset-generator/data/init_dataset.csv', index = None, header=True)
