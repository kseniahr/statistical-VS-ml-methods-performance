## This Class contains helping functions for running the simulation

# Import libaries
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import random

class SimulationModel:

    def simulate_next_populations(self, study_name, init_timestamp, defaults, coefficients, populations_collection):
        """
            Input: List of defaults, year of population t1,  list of coefficients, dictionary of populations, dictionary of samples from each population
            Output: Dictionary of t populations
        """

        year_key = init_timestamp

        # Loop simulates future populations for next t-1 years.
        for t in range (0, 4):

            prev_year_key = year_key
            year_key = year_key + 1

            # Generate exogene variables (including latent variables like prediction-error)
            if study_name == 'study1' or study_name == 'study_combined':
                independent_vars, error, Y = self.change_vars_distribution(defaults, coefficients[t], populations_collection[prev_year_key])

            else:
                independent_vars, error, Y = self.generate_variables(defaults, coefficients[t])

            # Combine dependent and independent variables in a data-frame
            populations_collection[year_key] = pd.DataFrame({'X1': independent_vars[0], 'X2': independent_vars[1], 'X3': independent_vars[2], 'Y': Y, 'error': error})

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
        X1 = scale(populations_collection['X1'] + np.random.normal(0.0, 1.0, defaults['n_rows']))
        X2 = scale(populations_collection['X2'] + np.random.normal(0.0, 1.0, defaults['n_rows']))
        X3 = scale(populations_collection['X3'] + np.random.normal(0.0, 1.0, defaults['n_rows']))

        error = np.random.normal(0.0, coefficients[4], defaults['n_rows'])

        # calculate endogene variables
        Y = coefficients[0] + coefficients[1]*X1 + coefficients[2]*X2 + coefficients[3]*X3 + error

        independent_vars = [X1, X2, X3]

        return independent_vars, error, Y


    # -------------------------------------------------

    def generate_variables(self, defaults, coefficients):
        """
            Description: Generates exogene variables (including latent variables like prediction-error)
            Input: List of defaults, list of coefficients
            Output: List of X1, X2, X3 values, error, dependent variable Y (target)
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

    def create_samples_collection(self, init_timestamp, defaults, populations_collection, samples_list_collection):
        """
            Input: List of defaults, year of population t1,  dictionary of populations, dictionary of samples from each population
            Output: Dictionary of samples for each population
        """

        year_key = init_timestamp

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
            samples_list[i].insert(0, 'sample_n', i)

        return samples_list

    # -------------------------------------------------
