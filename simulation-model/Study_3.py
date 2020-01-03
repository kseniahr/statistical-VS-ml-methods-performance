## This script evaluates the overtime performance if mean of the dependent variable beta_0 changes

# Import objects
from SimulationModel import SimulationModel
from Evaluation import Evaluation
import copy

class Study_3():

    def __init__(self, defaults, coefficients, df, mean_change, complexity):

        """
        Description: This method is called when an object is created from a
        class and it allows the class to initialize the attributes of the class
        Input: List of defaults, dataframe
        """
        self.defaults = defaults
        self.coefficients = coefficients
        self.df = df
        self.mean_change = mean_change
        self.complexity = complexity


    def create_target_mean(self):

        """
        Description: This method introduces unknown factor increase of target mean
        Input: none
        Output: Dictionary of coefficients for each year in the simulation
        """

        year = self.defaults['start_year'] + 1

        for i in range(self.defaults['n_years']-1):

            dict = self.coefficients[year-1]

            item_copy = copy.copy(dict)

            item_copy.update({'intercept': (item_copy['intercept'] + 1) * self.mean_change})

            self.coefficients.update({year: item_copy})

            year = year + 1

        return self.coefficients

    #-------------------------------------------------

    def run_simulation(self):
        """
        Description: This method evaluates model performance when
        mean of the dependent variable beta_0 changes overtime by 'mean_change'
        value
        Input: none
        Output: Plots of MSE evolution overtime
        """
        # Initialize empty dictionaries of accuracy metrics
        population_scores_mlr = {}
        population_scores_rfr = {}
        population_scores_gbr = {}

        self.coefficients = self.create_target_mean()

        # Initialize a collection of year1 population as a dictionaty
        populations_collection = {self.defaults['start_year'] : self.df}

        # Initialize an empty collection of samples
        samples_list_collection = {}

        simulation_obj = SimulationModel()

        populations_collection = simulation_obj.simulate_next_populations('study3', \
         self.defaults, self.coefficients, populations_collection, self.complexity)

        samples_list_collection = simulation_obj.create_samples_collection(self.defaults, \
         populations_collection, samples_list_collection)

        eval_obj = Evaluation()

        population_scores_mlr, population_scores_rfr, population_scores_gbr = eval_obj.train(self.defaults, \
         population_scores_mlr, population_scores_rfr, population_scores_gbr, samples_list_collection)

        # Now we create plots that visualize MSE of each model for a timespan of t years
        eval_obj.create_plot_MSE(self.defaults, population_scores_mlr, population_scores_rfr, \
         population_scores_gbr, 'Study 3: MSE overtime')
        
        print('Simulation of mean change of the dependent variable, including Linear Regression, \
        Random Forest Regression and Gradient Boosting Regression on '+ str(self.defaults['n_rows']) + ' artificially \
        generated observations for each of ' + str(self.defaults['n_years']) + ' years is finished.')