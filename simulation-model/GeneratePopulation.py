## This script creates a class which generates artificial dataset

# Import libraries
import sys
import numpy as np
import pandas as pd

class GeneratePopulation():

    def __init__(self, defaults):
        """
            Description: This method is called when an object is created from a class and it
            allows the class to initialize the attributes of the class
            Input: List of defaults
            Output: Artificially generated dataset saved into folder 'data'
        """

        coefficients_y1 = self.create_beta_coefs(defaults)

        # Generate exogene variables (including latent variables like prediction-error)
        independent_vars, error, Y = self.generate_population(defaults, coefficients_y1)

        # Combine dependent and independent variables in a data-frame

        population = pd.concat([pd.DataFrame(independent_vars), pd.DataFrame({'error': error, 'Y' : Y})], axis=1)

        # Export initial dataset for running the simulation
        population.to_csv('data/init_population.csv', index = None, header=True)

    #-------------------------------------------------

    def create_beta_coefs(self, defaults):
        """
            Description: This function generates a dictionary of b beta coefficients from a list of values [-1,1],
            where b is the number of independent variables predefined in the script 'Main' (defaults[5])
            Input: Number of independent variables count(X1, X2, ...Xb), dictionary of coefficients
            Output: dictionary of coefficient(intercept (beta0), beta coefficient)
        """
        # Initialize a dictionary with beta0 coefficient (intercept_y1) = 0 and error_y1 = 1
        coefficients_y1 = {'intercept_y1': 0, 'error_y1': 1}

        for i in range(defaults['num_X']):
            # create name for a key in a dictionary. For example, 'beta1_y1'
            beta = "beta" + str(i+1)+"_y1"
            # initialize value of a coefficient with either 1 or -1 (can be manually changed to any other values)
            coefficients_y1[beta] = np.random.choice(a = [1,-1])

        return coefficients_y1

    #-------------------------------------------------

    def generate_population(self, defaults, coefficients):
        """
            Description: Generates independent variables (including latent variables like prediction-error)
            Input: List of defaults, list of coefficients
            Output: dict of independent variables (X1, X2,...,Xb), prediction error and dependent variable Y
        """
        # Create empty dictionary of b independent variables (X1, X2, ..., Xb)
        independent_vars = {}

        # This for-loop creates normally distributed values for X1, X2, ..., Xb independent variables
        for i in range(defaults['num_X']):
            X = 'X'+ str(i+1)
            independent_vars[X] = np.random.normal(0.0, 1.0, defaults['num_rows'])

        # Assign normally distributed values to be an error
        error = np.random.normal(0.0, coefficients['error_y1'], defaults['num_rows'])

        Y = self.calculate_dependent_var(defaults, coefficients, independent_vars, error)

        return independent_vars, error, Y

    # -------------------------------------------------

    def calculate_dependent_var(self, defaults, coefficients, independent_vars, error):
        """
            Description: Calculate dependent variable Y (based on intercept coef, beta coef,
            independent variables and prediction error)
            Input: List of defaults, dict of coefficients, dict of independent variables
            (X1, X2,..., Xb) and prediction error
            Output: values of dependent variable Y
        """
        Y = coefficients['intercept_y1'] + error

        for i in range(defaults['num_rows']):
            beta_i = "beta" + str(i+1)+"_y1"
            X_i = 'X'+ str(i+1)

            Y = Y + coefficients[beta_i]*independent_vars[X_i]

        return Y

    # -------------------------------------------------
