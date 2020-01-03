## This script creates a class which generates artificial dataset

# Import libraries
import sys
import numpy as np
import pandas as pd
import random

class GeneratePopulation():

    def __init__(self, defaults):
        """
            Description: This method is called when an object is created from a
            class and it allows the class to initialize the attributes of the
            class
            Input: List of defaults
        """
        self.defaults = defaults

    #-------------------------------------------------

    def create_beta_coefs(self):
        """
            Description: This function generates a dictionary of b beta
            coefficients from a list of values [-1,1], where b is the number of
            independent variables predefined in the script 'Main' (defaults[5])
            Input: Number of independent variables count(X1, X2, ...Xb),
            dictionary of coefficients
            Output: dictionary of coefficient(intercept (beta0), beta coefficient)
        """
        # Initialize a dictionary with beta0 coefficient (intercept_y1) = 0 and error_y1 = 1
        coefficients_y1 = {'intercept': 0, 'error': 1}

        for i in range(self.defaults['n_X']):
            # create name for a key in a dictionary. For example, 'beta1_y1'
            beta = "beta" + str(i+1)
            # initialize value of a coefficient with either 1 or -1
            # can be manually changed to any other values
            coefficients_y1[beta] = np.random.choice(a = [1,-1])

        return coefficients_y1

    #-------------------------------------------------

    def generate_population(self, complexity):
        """
            Description: Artificially generated dataset saved into folder 'data'
            Input: none
            Output: list of beta coefficients
        """

        coefficients_y1 = self.create_beta_coefs()

        # Create empty dictionary of b independent variables (X1, X2, ..., Xb)
        independent_vars = {}

        # This for-loop creates normally distributed values for X1, X2, ..., Xb
        for i in range(self.defaults['n_X']):
            X = 'X'+ str(i+1)
            independent_vars[X] = np.random.normal(0.0, 1.0, self.defaults['n_rows'])

        # Assign normally distributed values to be an error
        random.seed(42)
        error = np.random.normal(0.0, coefficients_y1['error'], self.defaults['n_rows'])

        if complexity == 'linear':
            Y = self.calculate_dependent_var_linear(coefficients_y1, independent_vars, error)
        elif complexity == 'polynomial':
            Y = self.calculate_dependent_var_polynomial(coefficients_y1, independent_vars, error)
        else:
            print('This type of complexity does not exist.')

        # Combine dependent and independent variables in a data-frame
        population = pd.concat([pd.DataFrame(independent_vars), \
                        pd.DataFrame({'error': error, 'Y' : Y})], axis=1)


        # Export initial dataset for running the simulation
        population.to_csv('data/init_population_' + complexity + '.csv', index = None, header=True)

        return coefficients_y1

    # -------------------------------------------------

    def calculate_dependent_var_linear(self, coefficients, independent_vars, error):
        """
            Description: Calculate dependent variable Y as a linear combination(based
            on intercept coef, beta coef,
            independent variables and prediction error)
            Input: List of defaults, dict of coefficients, dict of independent
            variables (X1, X2,..., Xb) and prediction error
            Output: values of dependent variable Y
        """
        Y_linear = coefficients['intercept'] + error

        for j in range(self.defaults['n_X']):

            beta_i = "beta" + str(j+1)
            X_i = 'X'+ str(j+1)

            Y_linear = Y_linear + coefficients[beta_i]*independent_vars[X_i]

        return Y_linear

    # -------------------------------------------------

    def calculate_dependent_var_polynomial(self, coefficients, independent_vars, error):
        """
            Description: Calculate dependent variable Y as a linear combination(based
            on intercept coef, beta coef,
            independent variables and prediction error)
            Input: List of defaults, dict of coefficients, dict of independent
            variables s(X1, X2,..., Xb) and prediction error
            Output: values of dependent variable Y
        """
        Y_polynom = coefficients['intercept'] + error

        for j in range(self.defaults['n_X']):

            beta_i = "beta" + str(j+1)
            X_i = 'X'+ str(j+1)

            Y_polynom = Y_polynom + coefficients[beta_i] * independent_vars[X_i]

        return Y_polynom
