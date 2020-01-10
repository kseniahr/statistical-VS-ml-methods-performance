
# Import visualization libraries
import matplotlib.pylab as pl
import seaborn as sns

import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
# Import forecasting libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

import numpy as np
import pandas as pd

class Evaluation:

    # Define which models are going to be tested
    mlr = LinearRegression() # statistical model
    rfr  = RandomForestRegressor(n_estimators = 10, random_state = 42) # ML model
    gbr  = GradientBoostingRegressor() # ML model

    # -------------------------------------------------

    def train(self, defaults, population_scores_mlr, population_scores_rfr, \
     population_scores_gbr, samples_list_collection):

        year_key = defaults['start_year'] # year 2019

        mlr_model = {}
        rfr_model = {}
        gbr_model = {}
        # Fit all models to the samples of next t years
        for t in range (0, defaults['n_years']):

            population_scores_mlr[year_key], mlr_model =  self.fit_lr(defaults, \
             samples_list_collection[year_key], self.mlr, t, mlr_model)
            population_scores_rfr[year_key], rfr_model =  self.fit_rfr(defaults, \
             samples_list_collection[year_key], self.rfr, t, rfr_model)
            population_scores_gbr[year_key], gbr_model =  self.fit_gbr(defaults, \
             samples_list_collection[year_key], self.gbr, t, gbr_model)

            year_key = year_key + 1

        print(population_scores_mlr)
        print(population_scores_rfr)
        print(population_scores_gbr)
        return  population_scores_mlr, population_scores_rfr, population_scores_gbr

    # -------------------------------------------------
    # Fit LinearRegression to all samples of Y1, Y2, ..... Yt
    def fit_lr(self, defaults, samples_list, stat_model, year, mlr_model):

        # Define arrays of accuracy scores
        MSE_mlr   = np.empty(defaults['n_samples'])
        MAPE_mlr  = np.empty(defaults['n_samples'])
        sMAPE_mlr = np.empty(defaults['n_samples'])

        for i in range(0, defaults['n_samples']):

            X = samples_list[i].drop(['Y', 'error'], axis = 1) # here we have 3 variables for multiple regression.
            Y = samples_list[i]['Y']

            X_train, X_test, y_train, y_test = train_test_split(X, Y, \
             test_size=0.2, random_state=0)

            if year == 0:
                mlr_model[i] = stat_model.fit(X_train, y_train)

            y_pred = mlr_model[i].predict(X_test)

            MSE_mlr[i], MAPE_mlr[i], sMAPE_mlr[i] = self.calculate_perform_metrics(y_test, y_pred)

        # Record MSE, MAPE, sMAPE values of each sample in a DataFrame
        scores_mlr = {'MSE': round(MSE_mlr.mean(), 2), 'MAPE': MAPE_mlr.mean(), \
         'sMAPE': sMAPE_mlr.mean()}

        # Return average accuracy score value of k samples
        return scores_mlr, mlr_model



    # Fit RandomForestRegressor to all samples of Y1, Y2, ..... Yt (untuned)
    def fit_rfr(self, defaults, samples_list, ML_model, year, rfr_model):

        # Define arrays of accuracy scores
        MSE_rfr   = np.empty(defaults['n_samples'])
        MAPE_rfr  = np.empty(defaults['n_samples'])
        sMAPE_rfr = np.empty(defaults['n_samples'])

        for i in range(0, defaults['n_samples']):
            X = samples_list[i].drop(['Y', 'error'], axis = 1) # here we have 3 variables for multiple regression.
            Y = samples_list[i]['Y']

            X_train, X_test, y_train, y_test = train_test_split(X, Y, \
             test_size=0.2, random_state=0)

            if year == 0:
                rfr_model[i] = ML_model.fit(X_train, y_train)

            y_pred = rfr_model[i].predict(X_test)

            MSE_rfr[i], MAPE_rfr[i], sMAPE_rfr[i] = self.calculate_perform_metrics(y_test, y_pred)

        # Record MSE, MAPE, sMAPE values of each sample in a DataFrame
        scores_rfr = {'MSE': MSE_rfr.mean(), 'MAPE': MAPE_rfr.mean(), \
         'sMAPE': sMAPE_rfr.mean()}

        # Return average accuracy score value of k samples
        return scores_rfr, rfr_model

    # -------------------------------------------------

    # Fit GradientBoostingRegressor to all samples of Y1, Y2, ..... Yt
    def fit_gbr(self, defaults, samples_list, ML_model, year, gbr_model):

        # Define arrays of accuracy scores
        MSE_gbr   = np.empty(defaults['n_samples'])
        MAPE_gbr  = np.empty(defaults['n_samples'])
        sMAPE_gbr = np.empty(defaults['n_samples'])

        for i in range(0, defaults['n_samples']):
            X = samples_list[i].drop(['Y', 'error'], axis = 1) # here we have 3 variables for multiple regression.
            Y = samples_list[i]['Y']

            X_train, X_test, y_train, y_test = train_test_split(X, Y, \
             test_size=0.2, random_state=0)

            if year == 0:
                gbr_model[i] = ML_model.fit(X_train, y_train)

            y_pred = gbr_model[i].predict(X_test)

            MSE_gbr[i], MAPE_gbr[i], sMAPE_gbr[i] = self.calculate_perform_metrics(y_test, y_pred)

        # Record MSE, MAPE, sMAPE values of each sample in a DataFrame
        scores_gbr = {'MSE': MSE_gbr.mean(), 'MAPE': MAPE_gbr.mean(), \
         'sMAPE': sMAPE_gbr.mean()}

        # Return average accuracy score value of k samples
        return scores_gbr, gbr_model

    # -------------------------------------------------

    def calculate_perform_metrics(self, y_test, y_pred):
        # Calculate mean squarred error (MSE)
        MSE = metrics.mean_squared_error(y_test, y_pred)
        # Calculate mean absolute percentage error (MAPE)
        MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        # Calculate symmetric mean absolute percentage error (sMAPE)
        sMAPE = 100/len(y_test) * np.sum(2 * np.abs(y_pred - y_test) / (np.abs(y_test) \
         + np.abs(y_pred)))

        return MSE, MAPE, sMAPE

    # -------------------------------------------------

    # Creates a Figure that represents MSE score on y-axis and Year on x-y_axis
    def create_plot_MSE(self, defaults, population_scores_mlr, population_scores_rfr, \
     population_scores_gbr, name, dimensionality, complexity, var_type):


        list_x_axis = [x+2019 for x in range(defaults['n_years'])]

        list_y_axis_mlr = [population_scores_mlr[defaults['start_year']+x]['MSE'] \
         for x in range(defaults['n_years'])]
        list_y_axis_gbr = [population_scores_gbr[defaults['start_year']+x]['MSE'] \
         for x in range(defaults['n_years'])]
        list_y_axis_rfr = [population_scores_rfr[defaults['start_year']+x]['MSE'] \
         for x in range(defaults['n_years'])]



        # Create 3x1 sub plots
        gs = gridspec.GridSpec(3, 1)
        pl.figure(name)
        pl.rcParams['figure.dpi'] = 300

        ax = pl.subplot(gs[0, 0]) # row 2, col 0
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pl.plot(range(len(list_x_axis)), list_y_axis_mlr, color = 'black', linestyle = '-')
        pl.xticks(range(len(list_x_axis)),list_x_axis)
        pl.legend(['MLR'])
        pl.title('MultipleLinearRegression', fontsize = 9)
        pl.ylabel('MSE', fontsize = 9)
        pl.xlim(0, len(list_x_axis))
        pl.ylim(0, 10)

        ax = pl.subplot(gs[1, 0]) # row 0, col 0
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pl.plot(range(len(list_x_axis)), list_y_axis_gbr, color = 'black', linestyle = '--')
        pl.xticks(range(len(list_x_axis)),list_x_axis)
        pl.legend(['GBR'])
        pl.title('GradientBoostingRegressor', fontsize = 9)
        pl.ylabel('MSE', fontsize = 9)
        pl.xlim(0, len(list_x_axis))
        pl.ylim(0, 10)

        ax = pl.subplot(gs[2, 0]) # row 0, col 0
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pl.plot(range(len(list_x_axis)), list_y_axis_rfr, color = 'black', linestyle = '--')
        pl.xticks(range(len(list_x_axis)),list_x_axis)
        pl.legend(['RFR'])
        pl.title('RandomForestRegressor', fontsize = 9)
        pl.ylabel('MSE', fontsize = 9)
        pl.xlim(0, len(list_x_axis))
        pl.ylim(0, 10)

        pl.tight_layout()
        pl.savefig('plots/MSE_' + dimensionality + complexity + var_type + '_' + name + '.png')
        pl.show()

    # -------------------------------------------------

    # Creates a Figure of t histograms that represent the distribution of each year of feature X1
    def create_histograms(self, defaults, populations_collection, name, dimensionality, complexity, var_type):
        if defaults['n_years']<=5:
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

        else:
            gs = gridspec.GridSpec(2, int(defaults['n_years']/2))
            pl.rc('font', size = 6) # font of x and y axes
            pl.figure(name)

            year_key = defaults['start_year']

            for p in range(0, defaults['n_years']):
                if p < 5:
                    population = populations_collection[year_key]
                    year_key = year_key + 1
                    ax = pl.subplot(gs[0, p]) # row 0, col 0
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    pl.title('Year'+str(p+1), fontsize = 12)
                    pl.ylabel('Frequency')
                    pl.hist(population['X1'], color = 'grey', edgecolor = 'black', alpha=.3)
                else:
                    population = populations_collection[year_key]
                    year_key = year_key + 1
                    ax = pl.subplot(gs[1, p-int(defaults['n_years']/2)]) # row 0, col 0
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    pl.title('Year'+str(p+1), fontsize = 12)
                    pl.ylabel('Frequency')
                    pl.hist(population['X1'], color = 'grey', edgecolor = 'black', alpha=.3)

        pl.tight_layout() # add space between subplots
        pl.savefig('plots/Histograms_' + dimensionality + complexity + var_type + '_' + name + '.png')
        pl.show()
    # -------------------------------------------------

    # Creates a Figure of t heatmaps that represent the correlation between features
    def create_correlation_plots(self, defaults, populations_collection, name, dimensionality, complexity, var_type):

        year_key = defaults['start_year']
        # Basic correlogram
        for p in range(defaults['n_years']):
            df = populations_collection[year_key].drop('error', axis = 1)
            sns_plot = sns.pairplot(df)
            sns_plot.savefig('plots/Correlation_' + dimensionality + complexity + var_type + '_' + name + str(p+1)+  '.png')
            year_key = year_key + 1
    # -------------------------------------------------
