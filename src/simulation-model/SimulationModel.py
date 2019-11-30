# Import libaries
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import scale
# Import scripts with all functions
from functions import change_vars_distribution

#from main import parameters
#from train_init_dataset import df, samples_list, scores_mlr, scores_rfr, scores_gbr

class SimulationModel:

    def simulate_next_populations(self, study_name, init_timestamp, parameters, coefficients, populations_collection):
        """
            Input: List of parameters, year of population t1,  list of coefficients, dictionary of populations, dictionary of samples from each population
            Output: Dictionary of t populations
        """

        year_key = init_timestamp

        # Loop simulates future populations for next t years.
        for t in range (0, parameters[3]):

            prev_year_key = year_key
            year_key = year_key + 1

            # Generate exogene variables (including latent variables like prediction-error)
            if study_name == 'study1' or study_name == 'study_combined':
                independent_vars, error, Y = self.change_vars_distribution(parameters, coefficients[t], populations_collection[prev_year_key])

            else:
                independent_vars, error, Y = self.generate_variables(parameters, coefficients[t])

            # Combine dependent and independent variables in a data-frame
            populations_collection[year_key] = pd.DataFrame({'X1': independent_vars[0], 'X2': independent_vars[1], 'X3': independent_vars[2], 'Y': Y, 'error': error})

        return populations_collection

    # -------------------------------------------------

    def change_vars_distribution(self, parameters, coefficients, populations_collection):
        """
            Input: List of parameters, list of coefficients, dictionary of populations
            Output: List of indpendent variables (X1, X2, X3), error, dependent variable (Y)
        """
        # the arguments of np.random.normalnorm(mean, sd, number of random numbers N)
        # the scale function transforms the input values in a range from 0 to 1
        # I used a autoregressive function, meaning that t1 will influence the values on t2, t3 will influence t4 and so on
        X1 = scale(populations_collection['X1'] + np.random.normal(0.0, 1.0, parameters[0]))
        X2 = scale(populations_collection['X2'] + np.random.normal(0.0, 1.0, parameters[0]))
        X3 = scale(populations_collection['X3'] + np.random.normal(0.0, 1.0, parameters[0]))

        error = np.random.normal(0.0, coefficients[4], parameters[0])

        # calculate endogene variables
        Y = coefficients[0] + coefficients[1]*X1 + coefficients[2]*X2 + coefficients[3]*X3 + error

        independent_vars = [X1, X2, X3]

        return independent_vars, error, Y


    # -------------------------------------------------

    # Generate exogene variables (including latent variables like prediction-error)
    def generate_variables(self, parameters, coefficients):
        # seed is used to keep the same X1..Xn variables for each population
        np.random.seed(42)
        X1 = np.random.normal(0.0, 1.0, parameters[0])
        np.random.seed(43)
        X2 = np.random.normal(0.0, 1.0, parameters[0])
        np.random.seed(44)
        X3 = np.random.normal(0.0, 1.0, parameters[0])

        error = np.random.normal(0.0, coefficients[4], parameters[0])

        # calculate endogene variables
        Y = coefficients[0] + coefficients[1]*X1 + coefficients[2]*X2 + coefficients[3]*X3 + error

        independent_variables = [X1, X2, X3]

        return independent_variables, error, Y

    # -------------------------------------------------

    def create_samples_collection(self, init_timestamp, parameters, populations_collection, samples_list_collection):
        """
            Input: List of parameters, year of population t1,  dictionary of populations, dictionary of samples from each population
            Output: Dictionary of samples for each population
        """

        year_key = init_timestamp

        # Loop creates k samples for each year with n observations in each sample.
        for t in range (0, parameters[3]):

            prev_year_key = year_key
            year_key = year_key + 1

            samples_list_collection[year_key] = self.draw_k_samples(parameters, populations_collection[year_key])

        return samples_list_collection

    # -------------------------------------------------

    def draw_k_samples(self, parameters, df):
        """
            Input: List of parameters, population of year t
            Output: List of k samples for population of year t
        """
        # Define a list of k samples with value 0
        samples_list = [[0]]*parameters[2]         # parameters[2] = k
        # Create k samples with n observations using random.sample
        for i in range(0, parameters[2]):
            index = random.sample(range(0, parameters[0]), parameters[1])
            samples_list[i] = df.iloc[ index, :]
            samples_list[i].insert(0, 'sample_n', i)

        return samples_list
