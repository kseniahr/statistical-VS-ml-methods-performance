## This script evaluates the overtime performance if the distribution of input variables changes (X1,X2,X3)

# Import objects
from SimulationModel import SimulationModel
from Evaluation import Evaluation
import copy

class Study_1():

    def __init__(self, defaults, coefficients, df, beta_change, complexity):
        """
        Description: This method is called when an object is created from a
        class and it allows the class to initialize the attributes of the class
        Input: List of defaults, dataframe
        """
        self.defaults = defaults
        self.coefficients = coefficients
        self.df = df
        self.beta_change = beta_change
        self.complexity = complexity

    #-------------------------------------------------

    def create_beta_coefs(self):

        """
        Description: This method increases in beta1 coefficient overtime
        Input: none
        Output: Dictionary of coefficients for each year in the simulation
        """

        year = self.defaults['start_year'] + 1

        for i in range(self.defaults['n_years']-1):

            dict = self.coefficients[year-1]

            item_copy = copy.copy(dict)

            item_copy.update({'beta1': dict['beta1'] * self.beta_change})

            self.coefficients.update({year: item_copy})

            year = year + 1

        return self.coefficients
    #-------------------------------------------------

    def run_simulation(self):
        """
        Description: This method evaluates model performance when
        the distribution of input variables changes (X1,X2,X3) overtime
        Input: none
        Output: Plots of MSE evolution overtime
        """
        # Initialize empty dictionaries of accuracy metrics
        population_scores_mlr = {}
        population_scores_rfr = {}
        population_scores_gbr = {}

        self.coefficients = self.create_beta_coefs()

        # Initialize a collection of year1 population as a dictionaty
        populations_collection = {self.defaults['start_year'] : self.df}

        # Initialize an empty collection of samples
        samples_list_collection = {}

        simulation_obj = SimulationModel()

        populations_collection = simulation_obj.simulate_next_populations('study1', \
         self.defaults, self.coefficients, populations_collection, self.complexity)

        samples_list_collection = simulation_obj.create_samples_collection(self.defaults, \
         populations_collection, samples_list_collection)

        eval_obj = Evaluation()

        population_scores_mlr, population_scores_rfr, population_scores_gbr = eval_obj.train(self.defaults, \
         population_scores_mlr, population_scores_rfr, population_scores_gbr, samples_list_collection)

        # Now we create histograms that visualize the distribution of feature X1 changing overtime:
        eval_obj.create_histograms(self.defaults, populations_collection, 'Study 1: distribution of X1')

        # Now we create plots that visualize MSE of each model for a timespan of t years
        eval_obj.create_plot_MSE(self.defaults, population_scores_mlr, population_scores_rfr, \
         population_scores_gbr, 'Study 1: MSE overtime') 
        
        print('Simulation of distribution change of input variables, including Linear Regression, \
        Random Forest Regression and Gradient Boosting Regression on % 2d artificially \
        generated observations for each of % 2d years is finished.' %(self.defaults['n_rows'], \
                                                                        self.defaults['n_years']))