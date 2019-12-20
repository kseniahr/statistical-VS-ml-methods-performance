## This Class contains helping functions for running the simulation

# Import libaries
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import random

class SimulationModel:

    def simulate_next_populations(self, study_name, defaults, coefficients, populations_collection):
        """
            Input: List of defaults, year of population t1,  list of coefficients, dictionary of populations, dictionary of samples from each population
            Output: Dictionary of t populations
        """

        year_key = defaults['start_year']

        # Loop simulates future populations for next t-1 years.
        for t in range (0, defaults['n_years']-1):

            prev_year_key = year_key
            year_key = year_key + 1

            # Generate exogene variables (including latent variables like prediction-error)
            if study_name == 'study1' or study_name == 'study_combined':

                independent_vars, Y = self.change_vars_distribution(defaults, coefficients[year_key], populations_collection[prev_year_key])
                # Combine dependent and independent variables in a data-frame
                populations_collection[year_key] = pd.concat([pd.DataFrame(independent_vars), pd.DataFrame({'Y' : Y})], axis=1)

            else:
                independent_vars, Y = self.generate_variables(defaults, coefficients[year_key], populations_collection[prev_year_key])
                # Combine dependent and independent variables in a data-frame
                populations_collection[year_key] = pd.concat([independent_vars, pd.DataFrame({'Y' : Y})], axis=1)


        return populations_collection

    # -------------------------------------------------

    def change_vars_distribution(self, defaults, coefficients, populations_collection):
        """
            Input: List of defaults, list of coefficients, dictionary of populations
            Output: List of indpendent variables (X1, X2, X3), error, dependent variable (Y)
        """
        # the arguments of np.random.normalnorm(mean, sd, number of random numbers N)
        # the scale function transforms the input values in a range from 0 to 1
        # I used a autoregressive function, meaning that t1 will influence the values on t2, t3 will influence t4 and so on

        # Create empty dictionary of b independent variables (X1, X2, ..., Xb)
        independent_vars = {}

        for i in range(defaults['n_X']):

            X = 'X'+ str(i+1)
            independent_vars[X] = scale(populations_collection[X] + np.random.normal(0.0, 1.0, defaults['n_rows']))

        # Assign normally distributed values to be an error
        error = np.random.normal(0.0, coefficients['error'], defaults['n_rows'])

        Y = self.calculate_dependent_var(defaults, coefficients, independent_vars, error)

        return independent_vars, Y

    # -------------------------------------------------

    def calculate_dependent_var(self, defaults, coefficients, independent_vars, error):
        """
            Description: Calculate dependent variable Y (based on intercept coef, beta coef,
            independent variables and prediction error)
            Input: List of defaults, dict of coefficients, dict of independent variables
            (X1, X2,..., Xb) and prediction error
            Output: values of dependent variable Y
        """
        Y = coefficients['intercept'] + error

        for i in range(defaults['n_X']):
            beta_i = "beta" + str(i+1)
            X_i = 'X'+ str(i+1)

            Y = Y + coefficients[beta_i]*independent_vars[X_i]

        return Y

    # -------------------------------------------------

    def generate_variables(self, defaults, coefficients, populations_collection):
        """
            Description: Generates exogene variables (including latent variables like prediction-error)
            Input: List of defaults, list of coefficients
            Output: List of X1, X2, X3 values, error, dependent variable Y (target)
        """

        # Keep the same values for X1, X2, ..., Xb like in the initial population
        filter_col = [col for col in populations_collection if col.startswith('X')]

        independent_vars = populations_collection[filter_col]

        error = populations_collection['error']

        Y = self.generate_variables(self, coefficients[year_key], independent_vars, error)

        return independent_variables, Y

    # -------------------------------------------------

    def create_samples_collection(self, defaults, populations_collection, samples_list_collection):
        """
            Input: List of defaults, year of population t1,  dictionary of populations, dictionary of samples from each population
            Output: Dictionary of samples for each population
        """

        year_key = defaults['start_year']

        # Loop creates k samples for each year with n observations in each sample.
        for t in range (0, defaults['n_years']):

            samples_list_collection[year_key] = self.draw_k_samples(defaults, populations_collection[year_key])
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
