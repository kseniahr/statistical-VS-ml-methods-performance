## This script evaluates the overtime performance if relationship between input variables changes (B1,B2,B3)

# Import objects
from SimulationModel import SimulationModel
from Evaluation import Evaluation
import copy

class Study_2():

    def __init__(self, defaults, coefficients, df, relationship_term):

        """
        Description: This method is called when an object is created from a
        class and it allows the class to initialize the attributes of the class
        Input: List of defaults, dataframe
        """
        self.defaults = defaults
        self.coefficients = coefficients
        self.df = df
        self.relationship_term = relationship_term

    #-------------------------------------------------

    def create_beta_coefs(self):
        """
        Description: This method dencreases beta1 coefficient by 0.1 and
        increases b3 by 0.1 overtime
        Input: none
        Output: Dictionary of coefficients for each year in the simulation
        """
        year = self.defaults['start_year'] + 1

        for i in range(self.defaults['n_years'] - 1):

            # coefficients from the previous year get initialized as parameters for var 'year'
            dict = self.coefficients[year-1]

            item_copy = copy.copy(dict)

            # beta1 coefficient for the next year decreases ...
            # ... which leads to increase of beta3 coefficient
            item_copy.update({'beta1': dict['beta1'] - self.relationship_term, \
             'beta3': dict['beta3'] + self.relationship_term})

            self.coefficients.update({year: item_copy})

            year = year + 1

        return self.coefficients

    #-------------------------------------------------

    def run_simulation(self):
        """
        Description: This method evaluates model performance when
        relationship between input variables changes overtime
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

        populations_collection = simulation_obj.simulate_next_populations('study2', \
         self.defaults, self.coefficients, populations_collection)

        samples_list_collection = simulation_obj.create_samples_collection(self.defaults, \
         populations_collection, samples_list_collection)

        eval_obj = Evaluation()

        population_scores_mlr, population_scores_rfr, population_scores_gbr = eval_obj.train(self.defaults, \
         population_scores_mlr, population_scores_rfr, population_scores_gbr, samples_list_collection)

        # Now we create plots that visualize MSE of each model for a timespan of t years
        eval_obj.create_plot_MSE(self.defaults, population_scores_mlr, population_scores_rfr, \
         population_scores_gbr, 'Study 2: MSE overtime')
