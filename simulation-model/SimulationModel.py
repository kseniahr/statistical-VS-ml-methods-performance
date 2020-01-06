## This Class contains helping functions for running the simulation

# Import libaries
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import random

class SimulationModel:

    def simulate_next_populations(self, study_name, defaults, coefficients, \
     populations_collection, dimensionality, complexity, var_type):
        """
            Input: List of defaults, year of population t1,  list of coefficients,
            dictionary of populations, dictionary of samples from each population
            Output: Dictionary of t populations
        """
        year_key = defaults['start_year']

        # Loop simulates future populations for next t-1 years.
        for t in range (0, defaults['n_years']-1):

            prev_year_key = year_key
            year_key = year_key + 1

            # Generate exogene variables (including latent variables like prediction-error)
            if study_name == 'study1' or study_name == 'study4':

                independent_vars, error, Y = self.change_vars_distribution(defaults, \
                 coefficients[year_key], populations_collection[prev_year_key], complexity, var_type)
                # Combine dependent and independent variables in a data-frame
                populations_collection[year_key] = pd.concat([pd.DataFrame(independent_vars), \
                 pd.DataFrame({'error': error, 'Y' : Y})], axis=1)

            else:
                independent_vars, error, Y = self.generate_variables(defaults, \
                 coefficients[year_key], populations_collection[prev_year_key], complexity)
                # Combine dependent and independent variables in a data-frame
                populations_collection[year_key] = pd.concat([independent_vars, \
                 pd.DataFrame({'error': error, 'Y' : Y})], axis=1)
        return populations_collection

    # -------------------------------------------------

    def change_vars_distribution(self, defaults, coefficients, populations_collection, complexity, var_type):
        """
            Input: List of defaults, list of coefficients, dictionary of
            populations
            Output: List of indpendent variables (X1, X2, X3), error,
            dependent variable (Y)
        """
        # the arguments of np.random.normalnorm(mean, sd, number of random numbers N)
        # the scale function transforms the input values in a range from 0 to 1.
        # I used a autoregressive function, meaning that t1 will influence the
        # values on t2, t3 will influence t4 and so on

        # Create empty dictionary of b independent variables (X1, X2, ..., Xb)
        independent_vars = {}

        if var_type == 'C':
            # This for-loop scales normally distributed continuous values for X1, X2, ..., Xb
            for i in range(defaults['n_X']):
                X = 'X'+ str(i+1)
                independent_vars[X] = scale(populations_collection[X] + \
                 np.random.normal(0.0, 1.0, defaults['n_rows']))

        elif var_type == 'H':
            # This for-loop creates normally distributed values for X1, X2, ..., Xb where
            #  some vars are binary and some are continuous
            for i in range(0, int(defaults['n_X']/2)):
                X = 'X'+ str(i+1)
                independent_vars[X] = scale(populations_collection[X] + \
                 np.random.normal(0.0, 1.0, defaults['n_rows']))

            for i in range(int(defaults['n_X']/2), defaults['n_X']):
                X = 'X'+ str(i+1)
                independent_vars[X] = np.random.choice(a = [0, 1], size = (defaults['n_rows'],))

        else:
            print('This variable type is not included in the simulatiom model.')

        # Assign normally distributed values to be an error (same error overtime)
        #random.seed(42)
        error = np.random.normal(0.0, coefficients['error'], defaults['n_rows'])

        if complexity == 'L':
            Y = self.calculate_dependent_var_linear(defaults, coefficients, independent_vars, error)

        elif complexity == 'P':
            Y = self.calculate_dependent_var_polynomial(defaults, coefficients, independent_vars, error)

        else:
            print('This type of complexity does not exist.')

        return independent_vars, error, Y

    # -------------------------------------------------

    def calculate_dependent_var_linear(self, defaults, coefficients, independent_vars, error):
        """
            Description: Calculate dependent variable Y (based on intercept coef,
            beta coef, independent variables and prediction error)
            Input: List of defaults, dict of coefficients, dict of independent
            variables (X1, X2,..., Xb) and prediction error
            Output: values of dependent variable Y
        """
        Y_linear = coefficients['intercept'] + error

        for j in range(defaults['n_X']):

            beta_i = "beta" + str(j+1)
            X_i = 'X'+ str(j+1)

            Y_linear = Y_linear + coefficients[beta_i]*independent_vars[X_i]

        return Y_linear

    # -------------------------------------------------

    def calculate_dependent_var_polynomial(self, defaults, coefficients, independent_vars, error):
        """
            Description: Calculate dependent variable Y as a non-linear polynomial
            function (based on intercept coef, beta coef, independent variables and prediction error)
            Input: List of defaults, dict of coefficients, dict of independent
            variables s(X1, X2,..., Xb) and prediction error
            Output: values of dependent variable Y
        """
        Y_polynom = coefficients['intercept'] + error

        for j in range(defaults['n_X']):

            beta_i = "beta" + str(j+1)
            X_i = 'X'+ str(j+1)

            Y_polynom = Y_polynom + coefficients[beta_i] * independent_vars[X_i]

        return Y_polynom

    # -------------------------------------------------

    def generate_variables(self, defaults, coefficients, populations_collection, complexity):
        """
            Description: Generates exogene variables (including latent variables
            like prediction-error)
            Input: List of defaults, list of coefficients
            Output: List of X1, X2, X3 values, error, dependent variable Y (target)
        """
        independent_vars = {}

        # Keep the same values for X1, X2, ..., Xb like in the initial population
        filter_col = [col for col in populations_collection if col.startswith('X')]

        independent_vars = populations_collection[filter_col]

        error = np.random.normal(0.0, coefficients['error'], defaults['n_rows'])

        if complexity == 'L':
            Y = self.calculate_dependent_var_linear(defaults, coefficients, independent_vars, error)
        elif complexity == 'P':
            Y = self.calculate_dependent_var_polynomial(defaults, coefficients, independent_vars, error)
        else:
            print('This type of complexity does not exist.')

        return independent_vars, error, Y

    # -------------------------------------------------

    def create_samples_collection(self, defaults, populations_collection, \
     samples_list_collection):
        """
            Input: List of defaults, year of population t1,  dictionary of
            populations, dictionary of samples from each population
            Output: Dictionary of samples for each population
        """

        year_key = defaults['start_year']

        # Loop creates k samples for each year with n observations in each sample.
        for t in range (0, defaults['n_years']):

            samples_list_collection[year_key] = self.draw_k_samples(defaults, \
             populations_collection[year_key])
            year_key = year_key + 1

        return samples_list_collection

    # -------------------------------------------------

    def draw_k_samples(self, defaults, df):
        """
            Input: List of defaults, population of year t
            Output: List of k samples for population of year t
        """
        # Define a list of k samples with value 0
        samples_list = [[0]]*defaults['n_samples']

        # Create k samples with n observations using random.sample
        for i in range(0, defaults['n_samples']):
            index = random.sample(range(0, defaults['n_rows']), defaults['n_rows_sample'])
            samples_list[i] = df.iloc[ index, :]
            #samples_list[i].insert(0, 'sample_n', i)

        return samples_list

    # -------------------------------------------------
