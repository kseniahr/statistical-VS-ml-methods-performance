## This script creates a class which generates artificial dataset

# Import libraries
import sys
import numpy as np
import pandas as pd

class CreatePopulation():

    def __init__(self, defaults):
        """
            Description: This method is called when an object is created from a class and it
            allows the class to initialize the attributes of the class
            Input: List of defaults
            Output: Artificially generated dataset saved into folder 'data'
        """
        # Define regressions coefficients for the 1st year
        intercept_y1 =   0
        beta1_y1 =   1
        beta2_y1 =   1
        beta3_y1 = (-1)
        error_y1 =   1

        # Create a list of predefined coefficients for a year 2019
        coefficients_y1 = [intercept_y1, beta1_y1, beta2_y1, beta3_y1, error_y1]

        # Generate exogene variables (including latent variables like prediction-error)
        independent_vars, error, Y = self.generate_variables(defaults, coefficients_y1)

        # Combine dependent and independent variables in a data-frame
        df = pd.DataFrame({'X1': independent_vars[0], 'X2': independent_vars[1], 'X3': independent_vars[2], 'Y': Y, 'error': error})


        # Export initial dataset for running the simulation
        df.to_csv('data/init_dataset.csv', index = None, header=True)

    # -------------------------------------------------

    def generate_variables(self, defaults, coefficients):
        """
            Description: Generates exogene variables (including latent variables like prediction-error)
            Input: List of defaults, list of coefficients
            Output: Independent X1, X2, X3, dependent variable Y and error
        """
        # seed is used to keep the same X1..Xn variables for each population
        np.random.seed(42)
        X1 = np.random.normal(0.0, 1.0, defaults['n_rows'])

        np.random.seed(43)
        X2 = np.random.normal(0.0, 1.0, defaults['n_rows'])

        np.random.seed(44)
        X3 = np.random.normal(0.0, 1.0, defaults['n_rows'])

        error = np.random.normal(0.0, coefficients[4], defaults['n_rows'])

        # calculate endogene variables
        Y = coefficients[0] + coefficients[1]*X1 + coefficients[2]*X2 + coefficients[3]*X3 + error

        independent_variables = [X1, X2, X3]

        return independent_variables, error, Y

    # -------------------------------------------------
