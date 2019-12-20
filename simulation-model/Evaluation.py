
# Import visualization libraries
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
# Import forecasting libraries
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

    def train(self, defaults, population_scores_mlr, population_scores_rfr, population_scores_gbr, samples_list_collection):

        year_key = defaults['start_year'] # year 2019

        # Fit all models to the samples of next t years
        for t in range (0, defaults['n_years']):

            population_scores_mlr[year_key] =  self.fit_lr(defaults, samples_list_collection[year_key], self.mlr)
            population_scores_rfr[year_key] =  self.fit_rfr(defaults, samples_list_collection[year_key], self.rfr)
            population_scores_gbr[year_key] =  self.fit_gbr(defaults, samples_list_collection[year_key], self.gbr)

            year_key = year_key + 1

        return  population_scores_mlr, population_scores_rfr, population_scores_gbr

    # -------------------------------------------------
    # Fit LinearRegression to all samples of Y1, Y2, ..... Yt
    def fit_lr(self, defaults, samples_list, stat_model):

        # Define arrays of accuracy scores
        MSE_mlr   = np.empty(defaults['n_samples'])
        MAPE_mlr  = np.empty(defaults['n_samples'])
        sMAPE_mlr = np.empty(defaults['n_samples'])

        for i in range(0, defaults['n_samples']):
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
    def fit_rfr(self, defaults, samples_list, ML_model):

        # Define arrays of accuracy scores
        MSE_rfr   = np.empty(defaults['n_samples'])
        MAPE_rfr  = np.empty(defaults['n_samples'])
        sMAPE_rfr = np.empty(defaults['n_samples'])

        for i in range(0, defaults['n_samples']):
            X = samples_list[i].drop('Y', axis = 1) # here we have 3 variables for multiple regression.
            Y = samples_list[i]['Y']

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

            ML_model.fit(X_train, y_train) #training the algorithm
            y_pred = ML_model.predict(X_test)

            MSE_rfr[i], MAPE_rfr[i], sMAPE_rfr[i] = self.calculate_perform_metrics(y_test, y_pred)

        # Record MSE, MAPE, sMAPE values of each sample in a DataFrame
        scores_rf = pd.DataFrame({'MSE': MSE_rfr, 'MAPE': MAPE_rfr, 'sMAPE': sMAPE_rfr})

        # Return average accuracy score value of k samples
        return scores_rf.mean()

    # -------------------------------------------------

    # Fit GradientBoostingRegressor to all samples of Y1, Y2, ..... Yt
    def fit_gbr(self, defaults, samples_list, ML_model):

        # Define arrays of accuracy scores
        MSE_gbr   = np.empty(defaults['n_samples'])
        MAPE_gbr  = np.empty(defaults['n_samples'])
        sMAPE_gbr = np.empty(defaults['n_samples'])

        for i in range(0, defaults['n_samples']):
            X = samples_list[i].drop('Y', axis = 1) # here we have 3 variables for multiple regression.
            Y = samples_list[i]['Y']

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

            ML_model.fit(X_train, y_train) #training the algorithm
            y_pred = ML_model.predict(X_test)

            MSE_gbr[i], MAPE_gbr[i], sMAPE_gbr[i] = self.calculate_perform_metrics(y_test, y_pred)

        # Record MSE, MAPE, sMAPE values of each sample in a DataFrame
        scores_gb = pd.DataFrame({'MSE': MSE_gbr, 'MAPE': MAPE_gbr, 'sMAPE': sMAPE_gbr})

        # Return average accuracy score value of k samples
        return scores_gb.mean()

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
    def create_plot_MSE(self, defaults, population_scores_mlr, population_scores_rfr, population_scores_gbr, name):

        #for i in range(defaults['n_years']):
        list_x_axis = [x+1 for x in range(defaults['n_years'])]
        list_y_axis_gbr = [population_scores_gbr[defaults['start_year']+x][0] for x in range(defaults['n_years'])]
        list_y_axis_rfr = [population_scores_rfr[defaults['start_year']+x][0] for x in range(defaults['n_years'])]
        list_y_axis_mlr = [population_scores_mlr[defaults['start_year']+x][0] for x in range(defaults['n_years'])]

        # Create 3x1 sub plots
        gs = gridspec.GridSpec(3, 1)
        pl.figure(name)

        ax = pl.subplot(gs[0, 0]) # row 0, col 0
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pl.plot(list_x_axis,list_y_axis_gbr)
        pl.title('GradientBoostingRegressor', fontsize = 12)
        pl.ylabel('MSE', fontsize = 9)
        pl.ylim(0, 15)

        ax = pl.subplot(gs[1, 0]) # row 1, col 0
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pl.plot(list_x_axis, list_y_axis_rfr)
        pl.title('RandomForestRegressor', fontsize = 12)
        pl.ylabel('MSE', fontsize = 9)
        pl.ylim(0, 15)

        ax = pl.subplot(gs[2, 0]) # row 2, col 0
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pl.plot(list_x_axis, list_y_axis_mlr)
        pl.title('LinearRegression', fontsize = 12)
        pl.ylabel('MSE', fontsize = 9)
        pl.xlabel('Year')
        pl.ylim(0, 15)

        pl.tight_layout()
        pl.show()


    # -------------------------------------------------

    # Creates a Figure of t histograms that represent the distribution of each year of feature X1
    def create_histograms(self, defaults, populations_collection, name):
        gs = gridspec.GridSpec(1, defaults['n_years'])
        pl.rc('font', size = 6) # font of x and y axes
        pl.figure(name)

        year_key = defaults['start_year']

        for p in range(0,defaults['n_years']):

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
    def create_correlation_plots(self, defaults, populations_collection, name):
        gs = gridspec.GridSpec(1, 5)
        pl.rc('font', size = 6) # font of x and y axes
        pl.figure(name)

        year_key = defaults['start_year']

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
