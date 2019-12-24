## This script evaluates the overtime performance if relationship between input variables changes (B1,B2,B3)

# Import objects
from SimulationModel import SimulationModel
from Evaluation import Evaluation

class Study2():

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
        for i in range(self.defaults['n_years']):

            year = self.defaults['start_year'] + (i+1)

            # coefficients from the previous year get initialized as a parameters for the next year
            self.coefficients[year] = self.coefficients[year-1]

            # beta1 coefficient for the next year decreases ...
            self.coefficients[year]['beta1'] = self.coefficients[year]['beta1'] \
             - self.relationship_term
            # ... which leads to increase of beta3 coefficient
            self.coefficients[year]['beta3'] = self.coefficients[year]['beta3'] \
             + self.relationship_term

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


        #
        # init_timestamp = defaults['start_year']
        #
        # # Define regressions coefficients for years 2 to 5
        # intercept_y2                 = 0
        # beta1_y2, beta2_y2, beta3_y2 = 0.9, 1, 1.1
        # error_sd_y2                  = 1
        # coefficients_y2 = [intercept_y2, beta1_y2, beta2_y2, beta3_y2, error_sd_y2]
        # #-------------------------------
        # intercept_y3                 = 0
        # beta1_y3, beta2_y3, beta3_y3 = 0.7, 1, 1.3
        # error_sd_y3                  = 1
        # coefficients_y3 = [intercept_y3, beta1_y3, beta2_y3, beta3_y3, error_sd_y3]
        # #-------------------------------
        # intercept_y4                 = 0
        # beta1_y4, beta2_y4, beta3_y4 = 0.5, 1, 1.5
        # error_sd_y4                  = 1
        # coefficients_y4 = [intercept_y4, beta1_y4, beta2_y4, beta3_y4, error_sd_y4]
        # #-------------------------------
        # intercept_y5                 = 0
        # beta1_y5, beta2_y5, beta3_y5 = 0.3 , 1, 1.7
        # error_sd_y5                  = 1
        # coefficients_y5 = [intercept_y5, beta1_y5, beta2_y5, beta3_y5, error_sd_y5]
        #
        # #-------------------------------------------------------------------------------
        #
        # # Initialize a list of coefficients for each year
        # coefficients = [coefficients_y2, coefficients_y3, coefficients_y4, coefficients_y5]
        #
        #
        # # Initialize empty dictionaries of accuracy metrics
        # population_scores_mlr = {}
        # population_scores_rfr = {}
        # population_scores_gbr = {}
        #
        # # Initialize a collection of year1 population as a dictionaty
        # populations_collection = {init_timestamp : df}
        #
        # # Initialize an empty collection of samples
        # samples_list_collection = {}
        #
        # simulation_obj = SimulationModel()
        #
        # populations_collection = simulation_obj.simulate_next_populations('study2', init_timestamp, defaults, coefficients, populations_collection)
        #
        # samples_list_collection = simulation_obj.create_samples_collection(init_timestamp, defaults, populations_collection, samples_list_collection)
        #
        # eval_obj = Evaluation()
        #
        # population_scores_mlr, population_scores_rfr, population_scores_gbr = eval_obj.train(init_timestamp, defaults, population_scores_mlr, population_scores_rfr, population_scores_gbr, samples_list_collection)
        #
        # # Now we create plots that visualize MSE of each model for a timespan of t years
        # eval_obj.create_plot_MSE(init_timestamp, population_scores_mlr, population_scores_rfr, population_scores_gbr, 'Study 2: MSE overtime')
