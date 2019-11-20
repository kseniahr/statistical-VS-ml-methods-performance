import pandas as pd
import numpy as np
import random


#---------------------------------------------------------------------------------------------

# Generate exogene variables (including latent variables like prediction-error)
def generate_dataset(parameters, coefficients):
    # np.random.normal draws random numbers from a normal distribution,
    # the arguments of np.random.normalnorm(mean, sd, number of random numbers)
    X1 = np.random.normal(0.0, 1.0, parameters[0])
    X2 = np.random.normal(0.0, 1.0, parameters[0])
    X3 = np.random.normal(0.0, 1.0, parameters[0])
    error = np.random.normal(0.0, coefficients[4], parameters[0])

    # calculate endogene variables
    Y = coefficients[0] + coefficients[1]*X1 + coefficients[2]*X2 + coefficients[3]*X3 + error

    # combine dependent and independent variables in a data-frame
    population = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'Y': Y, 'error': error})

    return population

#---------------------------------------------------------------------------------------------

def draw_k_samples(parameters, df):
    # Define a list of k samples with value 0
    samples_list = [[0]]*k
    # Create k samples with n observations using random.sample
    for i in range(0, parameters[2]):
        index = random.sample(range(0, parameters[0]), parameters[1])
        samples_list[i] = df.iloc[ index, :]
        samples_list[i].insert(0, 'sample_n', i)

    return samples_list

#---------------------------------------------------------------------------------------------
