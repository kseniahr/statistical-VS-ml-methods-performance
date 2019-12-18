
#Import visualization libraries
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

import numpy as np
import pandas as pd

class Evaluation:

    # Define which models are going to be tested
    mlr = LinearRegression() # statistical model
    rfr  = RandomForestRegressor() # ML model
    gbr  = GradientBoostingRegressor() # ML model

    # -------------------------------------------------

    def train(self, init_timestamp, parameters, population_scores_mlr, population_scores_rfr, population_scores_gbr, samples_list_collection):

        year_key = init_timestamp # year 2019

        # Fit all models to the samples of next t years
        for t in range (0, parameters['n_years']):

            population_scores_mlr[year_key] =  self.fit_lr(parameters, samples_list_collection[year_key], self.mlr)
            population_scores_rfr[year_key] =  self.fit_rfr(parameters, samples_list_collection[year_key], self.rfr)
            population_scores_gbr[year_key] =  self.fit_gbr(parameters, samples_list_collection[year_key], self.gbr)

            year_key = year_key + 1

        return  population_scores_mlr, population_scores_rfr, population_scores_gbr

    # -------------------------------------------------
    # Fit LinearRegression to all samples of Y1, Y2, ..... Yt
    def fit_lr(self, parameters, samples_list, stat_model):

        # Define arrays of accuracy scores
        MSE_mlr   = np.empty(parameters['n_samples'])
        MAPE_mlr  = np.empty(parameters['n_samples'])
        sMAPE_mlr = np.empty(parameters['n_samples'])

        for i in range(0, parameters['n_samples']):
            samples_list[i] = pd.DataFrame(samples_list[i], columns = ['X1' , 'X2', 'X3', 'Y'])
            X = samples_list[i].drop('Y', axis = 1) # here we have 3 variables for multiple regression.
            Y = samples_list[i]['Y']

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

            stat_model.fit(X_train, y_train) #training the algorithm
            y_pred = stat_model.predict(X_test)

            MSE_mlr[i], MAPE_mlr[i], sMAPE_mlr[i] = self.calculate_perform_metrics(y_test, y_pred)

        # Record MSE, MAPE, sMAPE values of each sample in a DataFrame
        scores_mlr = pd.DataFrame({'MSE': MSE_mlr, 'MAPE': MAPE_mlr, 'sMAPE': sMAPE_mlr})

        # Return average accuracy score value of k samples
        return scores_mlr.mean()

    # -------------------------------------------------

    # Fit RandomForestRegressor to all samples of Y1, Y2, ..... Yt
    def fit_rfr(self, parameters, samples_list, ML_model):

        # Define arrays of accuracy scores
        MSE_mlr   = np.empty(parameters['n_samples'])
        MAPE_mlr  = np.empty(parameters['n_samples'])
        sMAPE_mlr = np.empty(parameters['n_samples'])

        for i in range(0, parameters['n_samples']):

            samples_list[i] = pd.DataFrame(samples_list[i], columns = ['X1' , 'X2', 'X3', 'Y'])
            X = samples_list[i].drop('Y', axis = 1) # here we have 3 variables for multiple regression.
            Y = samples_list[i]['Y']

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

            ML_model.fit(X_train, y_train) #training the algorithm
            y_pred = ML_model.predict(X_test)

            MSE_mlr[i], MAPE_mlr[i], sMAPE_mlr[i] = self.calculate_perform_metrics(y_test, y_pred)

        # Record MSE, MAPE, sMAPE values of each sample in a DataFrame
        scores_rf = pd.DataFrame({'MSE': MSE_mlr, 'MAPE': MAPE_mlr, 'sMAPE': sMAPE_mlr})

        # Return average accuracy score value of k samples
        return scores_rf.mean()

    # -------------------------------------------------

    # Fit GradientBoostingRegressor to all samples of Y1, Y2, ..... Yt
    def fit_gbr(self, parameters, samples_list, ML_model):

        # Define arrays of accuracy scores
        MSE_mlr   = np.empty(parameters['n_samples'])
        MAPE_mlr  = np.empty(parameters['n_samples'])
        sMAPE_mlr = np.empty(parameters['n_samples'])

        for i in range(0, parameters['n_samples']):
            samples_list[i] = pd.DataFrame(samples_list[i], columns = ['X1' , 'X2', 'X3', 'Y'])
            X = samples_list[i].drop('Y', axis = 1) # here we have 3 variables for multiple regression.
            Y = samples_list[i]['Y']

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

            ML_model.fit(X_train, y_train) #training the algorithm
            y_pred = ML_model.predict(X_test)

            MSE_mlr[i], MAPE_mlr[i], sMAPE_mlr[i] = self.calculate_perform_metrics(y_test, y_pred)

        # Record MSE, MAPE, sMAPE values of each sample in a DataFrame
        scores_rf = pd.DataFrame({'MSE': MSE_mlr, 'MAPE': MAPE_mlr, 'sMAPE': sMAPE_mlr})

        # Return average accuracy score value of k samples
        return scores_rf.mean()

# -------------------------------------------------

    def calculate_perform_metrics(self, y_test, y_pred):
        # Calculate mean squarred error (MSE)
        MSE = metrics.mean_squared_error(y_test, y_pred)
        # Calculate mean absolute percentage error (MAPE)
        MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        # Calculate symmetric mean absolute percentage error (sMAPE)
        sMAPE = 100/len(y_test) * np.sum(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred)))

        return MSE, MAPE, sMAPE

# -------------------------------------------------

    # Creates a Figure that represents MSE score on y-axis and Year on x-y_axis

    def create_plot_MSE(self, init_timestamp, population_scores_mlr, population_scores_rfr, population_scores_gbr, name):
        # Create 3x1 sub plots
        gs = gridspec.GridSpec(3, 1)
        pl.figure(name)

        ax = pl.subplot(gs[0, 0]) # row 0, col 0
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pl.plot([1,2,3,4,5],[population_scores_gbr[init_timestamp][0], population_scores_gbr[init_timestamp+1][0],population_scores_gbr[init_timestamp+2][0],population_scores_gbr[init_timestamp+3][0], population_scores_gbr[init_timestamp+4][0]])
        pl.title('GradientBoostingRegressor', fontsize = 12)
        pl.ylabel('MSE', fontsize = 9)
        pl.ylim(0, 4)

        ax = pl.subplot(gs[1, 0]) # row 1, col 0
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pl.plot([1,2,3,4,5],[population_scores_rfr[init_timestamp][0],population_scores_rfr[init_timestamp+1][0],population_scores_rfr[init_timestamp+2][0],population_scores_rfr[init_timestamp+3][0], population_scores_rfr[init_timestamp+4][0]])
        pl.title('RandomForestRegressor', fontsize = 12)
        pl.ylabel('MSE', fontsize = 9)
        pl.ylim(0, 4)

        ax = pl.subplot(gs[2, 0]) # row 2, col 0
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pl.plot([1,2,3,4,5],[population_scores_mlr[init_timestamp][0],population_scores_mlr[init_timestamp+1][0],population_scores_mlr[init_timestamp+2][0],population_scores_mlr[init_timestamp+3][0], population_scores_mlr[init_timestamp+4][0]])
        pl.title('LinearRegression', fontsize = 12)
        pl.ylabel('MSE', fontsize = 9)
        pl.xlabel('Year')
        pl.ylim(0, 4)

        pl.tight_layout()
        pl.show()


    # -------------------------------------------------

    # Creates a Figure of t histograms that represent the distribution of each year of feature X1
    def create_histograms(self, init_timestamp, populations_collection, name):
        gs = gridspec.GridSpec(1, 5)
        pl.rc('font', size = 6) # font of x and y axes
        pl.figure(name)

        year_key = init_timestamp

        for p in range(0,5):

            population = populations_collection[year_key]
            year_key = year_key + 1
            ax = pl.subplot(gs[0, p]) # row 0, col 0
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            pl.title('Year'+str(p+1), fontsize = 12)
            pl.ylabel('Frequency')
            pl.hist(population['X1'], color = 'grey', edgecolor = 'black', alpha=.3)
        pl.tight_layout() # add space between subplots
        pl.show()


    # -------------------------------------------------

    # Creates a Figure of t heatmaps that represent the correlation between features
    def create_correlation_plots(self, init_timestamp, populations_collection, name):
        gs = gridspec.GridSpec(1, 5)
        pl.rc('font', size = 6) # font of x and y axes
        pl.figure(name)

        year_key = init_timestamp

        for p in range(0,5):

            population = populations_collection[year_key]
            year_key = year_key + 1
            corrs = population.corr()
            ax = pl.subplot(gs[0, p]) # row 0, col 0
            pl.title('Correlation Matrix Year'+str(p+1), fontsize = 12)
            pl.ylabel('Frequency')
            pl.matshow(corrs)
        pl.tight_layout() # add space between subplots
        pl.show()

    # -------------------------------------------------
