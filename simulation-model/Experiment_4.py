## This script evaluates the overtime performance of all 3 studies combined

# Import objects
from SimulationModel import SimulationModel
from Evaluation import Evaluation
import copy

class Experiment_4():

    def __init__(self, defaults, coefficients, df, beta_change, relationship_term, mean_change, dimensionality, complexity, var_type):

        """
        Description: This method is called when an object is created from a
        class and it allows the class to initialize the attributes of the class
        Input: List of defaults, dataframe
        """
        self.defaults = defaults
        self.coefficients = coefficients
        self.df = df
        self.beta_change = beta_change
        self.relationship_term = relationship_term
        self.mean_change = mean_change
        self.dimensionality = dimensionality
        self.complexity = complexity
        self.var_type = var_type

    def create_concept_drift(self):
        """
        Description: This method introduces unknown factor increase of target mean
        Input: none
        Output: Dictionary of coefficients for each year in the simulation
        """

        year = self.defaults['start_year'] + 1

        for i in range(self.defaults['n_years']-1):

            dict = self.coefficients[year-1]

            item_copy = copy.copy(dict)

            item_copy.update({'beta1': dict['beta1'] * self.beta_change})

            item_copy.update({'intercept': dict['intercept'] + self.mean_change, \
             'beta1': dict['beta1'] - self.relationship_term, \
              'beta3': dict['beta3'] + self.relationship_term \
               })

            self.coefficients.update({year: item_copy})

            year = year + 1

        return self.coefficients

    #-------------------------------------------------

    def run_simulation(self):
        """
        Description: This method evaluates model performance when
        concept drift is introduced
        Input: none
        Output: Plots of MSE evolution overtime
        """
        # Initialize empty dictionaries of accuracy metrics
        population_scores_mlr = {}
        population_scores_rfr = {}
        population_scores_gbr = {}

        self.coefficients = self.create_concept_drift()

        # Initialize a collection of year1 population as a dictionaty
        populations_collection = {self.defaults['start_year'] : self.df}

        # Initialize an empty collection of samples
        samples_list_collection = {}

        simulation_obj = SimulationModel()

        populations_collection = simulation_obj.simulate_next_populations('Experiment4', \
         self.defaults, self.coefficients, populations_collection, self.dimensionality, self.complexity, self.var_type)

        samples_list_collection = simulation_obj.create_samples_collection(self.defaults, \
         populations_collection, samples_list_collection)

        eval_obj = Evaluation()

        eval_obj.create_correlation_plots(self.defaults, populations_collection, 'Experiment 4: Corr', self.dimensionality, self.complexity, self.var_type)
        population_scores_mlr, population_scores_rfr, population_scores_gbr = eval_obj.train(self.defaults, \
         population_scores_mlr, population_scores_rfr, population_scores_gbr, samples_list_collection)

        # Now we create histograms that visualize the distribution of feature X1 changing overtime:
        eval_obj.create_histograms(self.defaults, populations_collection, 'Experiment 4: distribution of X1', self.dimensionality, self.complexity, self.var_type)

        # Now we create plots that visualize MSE of each model for a timespan of t years
        eval_obj.create_plot_MSE(self.defaults, population_scores_mlr, population_scores_rfr, \
         population_scores_gbr, 'Experiment 4 concept drift: MSE overtime', self.dimensionality, self.complexity, self.var_type)

        print('Simulation of Concept Drift, including Linear Regression, \
        Random Forest Regression and Gradient Boosting Regression on % 2d artificially \
        generated observations for each of % 2d years is finished.' %(self.defaults['n_rows'], \
                                                                        self.defaults['n_years']))
