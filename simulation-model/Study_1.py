## This script evaluates the overtime performance if the distribution of input variables changes (X1,X2,X3)

# Import objects
from SimulationModel import SimulationModel
from Evaluation import Evaluation

class Study_1():

    def __init__(self, defaults, coefficients, df, beta_change):
        """
        Description: This method is called when an object is created from a class and it
        allows the class to initialize the attributes of the class
        Input: List of defaults, dataframe
        """
        self.defaults = defaults
        self.coefficients = coefficients
        self.df = df
        self.beta_change = beta_change

    #-------------------------------------------------

    def create_beta_coefs(self):
        """
        Description: This method increases in beta1 coefficient overtime
        Input: none
        Output: Dictionary of coefficients for each year in the simulation
        """
        for i in range(self.defaults['n_years']):

            year = self.defaults['start_year'] + (i+1)

            # coefficients from the previous year get initialized as a parameters for the next year
            self.coefficients[year] = self.coefficients[year-1]

            # beta1 coefficient for the next year increases by 'beta_change'
            self.coefficients[year]['beta1'] = self.coefficients[year]['beta1'] * self.beta_change

        return self.coefficients

    #-------------------------------------------------

    def run_simulation(self):

        # Initialize empty dictionaries of accuracy metrics
        population_scores_mlr = {}
        population_scores_rfr = {}
        population_scores_gbr = {}

        # Initialize a collection of year1 population as a dictionaty
        populations_collection = {self.defaults['start_year'] : self.df}

        # Initialize an empty collection of samples
        samples_list_collection = {}

        simulation_obj = SimulationModel()

        populations_collection = simulation_obj.simulate_next_populations('study1', self.defaults, self.coefficients, populations_collection)

        samples_list_collection = simulation_obj.create_samples_collection(self.defaults, populations_collection, samples_list_collection)

        eval_obj = Evaluation()

        population_scores_mlr, population_scores_rfr, population_scores_gbr = eval_obj.train(self.defaults, population_scores_mlr, population_scores_rfr, population_scores_gbr, samples_list_collection)


        # # Now we create histograms that visualize the distribution of feature X1 changing overtime:
        eval_obj.create_histograms(self.defaults, populations_collection, 'Study 1: distribution of X1')
        #
        # # Now we create plots that visualize MSE of each model for a timespan of t years
        eval_obj.create_plot_MSE(self.defaults, population_scores_mlr, population_scores_rfr, population_scores_gbr, 'Study 1: MSE overtime')
