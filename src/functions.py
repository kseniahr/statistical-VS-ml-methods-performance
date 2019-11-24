import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.model_selection import train_test_split
# Import visualization libraries
import numpy as np
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# Call np.random.seed() to make examples with (pseudo) random data reproducible:
np.random.seed(1234)


def build_regression_dataset():
    from sklearn.datasets import make_regression
    # Generate a random regression problem
    X, y = make_regression(n_samples=1000, n_features=3, n_informative=2,
                           n_targets=1, random_state=100, noise=0.05)
    #y *= 0.01

    return pd.DataFrame({'X1': X[:,0], 'X2': X[:,1], 'X3': X[:,2], 'Y': y})

#---------------------------------------------------------------------------------------------

# Generate exogene variables (including latent variables like prediction-error)
def generate_vars(parameters, coefficients, independent_vars, year):

    # np.random.normal draws random numbers from a normal distribution,
    # the arguments of np.random.normalnorm(mean, sd, number of random numbers)
    # the scale function transforms the input values in a range from 0 to 1
    # I used a autoregressive function, meaning that t1 will influence the values on t2, t3 will influence t4 and so on
    if year == 'year1':
        X1 = np.random.normal(0.0, 1.0, parameters[0])
        X2 = np.random.normal(0.0, 1.0, parameters[0])
        X3 = np.random.normal(0.0, 1.0, parameters[0])
    else:
        X1 = scale(independent_vars['X1'] + np.random.normal(0.0, 1.0, parameters[0]))
        X2 = scale(independent_vars['X2'] + np.random.normal(0.0, 1.0, parameters[0]))
        X3 = scale(independent_vars['X3'] + np.random.normal(0.0, 1.0, parameters[0]))

    error = np.random.normal(0.0, coefficients[4], parameters[0])

    # calculate endogene variables
    Y = coefficients[0] + coefficients[1]*X1 + coefficients[2]*X2 + coefficients[3]*X3 + error

    independent_vars = [X1, X2, X3]

    return independent_vars, error, Y

#---------------------------------------------------------------------------------------------

def draw_k_samples(parameters, df):
    # Define a list of k samples with value 0
    samples_list = [[0]]*parameters[2]         # parameters[2] = k
    # Create k samples with n observations using random.sample
    for i in range(0, parameters[2]):
        index = random.sample(range(0, parameters[0]), parameters[1])
        samples_list[i] = df.iloc[ index, :]
        samples_list[i].insert(0, 'sample_n', i)

    return samples_list

#---------------------------------------------------------------------------------------------

def calculate_perform_metrics(y_test, y_pred):
    # Calculate mean squarred error (MSE)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    # Calculate mean absolute percentage error (MAPE)
    MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    # Calculate symmetric mean absolute percentage error (sMAPE)
    sMAPE = 100/len(y_test) * np.sum(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred)))

    return MSE, MAPE, sMAPE

#----------------------------------------------------------------------------------------------
# Fit Linear Regression to all samples of Y1, Y2, ..... Yt
#----------------------------------------------------------------------------------------------

def fit_lr(parameters, samples_list, stat_model):
    # Define arrays of accuracy scores
    MSE_mlr   = np.empty(parameters[2])
    MAPE_mlr  = np.empty(parameters[2])
    sMAPE_mlr = np.empty(parameters[2])

    for i in range(0, parameters[2]):
        samples_list[i] = pd.DataFrame(samples_list[i], columns = ['X1' , 'X2', 'X3', 'Y'])
        X = samples_list[i].drop('Y', axis = 1) # here we have 3 variables for multiple regression.
        Y = samples_list[i]['Y']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        stat_model.fit(X_train, y_train) #training the algorithm
        y_pred = stat_model.predict(X_test)

        MSE_mlr[i], MAPE_mlr[i], sMAPE_mlr[i] = calculate_perform_metrics(y_test, y_pred)

    # Record MSE, MAPE, sMAPE values of each sample in a DataFrame
    scores_mlr = pd.DataFrame({'MSE': MSE_mlr, 'MAPE': MAPE_mlr, 'sMAPE': sMAPE_mlr})

    # Return average accuracy score value of k samples
    return scores_mlr.mean()

#----------------------------------------------------------------------------------------------
# Fit RandomForestRegressor to all samples of Y1, Y2, ..... Yt
#----------------------------------------------------------------------------------------------

def fit_rfr(parameters, samples_list, ML_model):
    # Define arrays of accuracy scores
    MSE_mlr   = np.empty(parameters[2])
    MAPE_mlr  = np.empty(parameters[2])
    sMAPE_mlr = np.empty(parameters[2])

    for i in range(0, parameters[2]):
        samples_list[i] = pd.DataFrame(samples_list[i], columns = ['X1' , 'X2', 'X3', 'Y'])
        X = samples_list[i].drop('Y', axis = 1) # here we have 3 variables for multiple regression.
        Y = samples_list[i]['Y']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        ML_model.fit(X_train, y_train) #training the algorithm
        y_pred = ML_model.predict(X_test)

        MSE_mlr[i], MAPE_mlr[i], sMAPE_mlr[i] = calculate_perform_metrics(y_test, y_pred)

    # Record MSE, MAPE, sMAPE values of each sample in a DataFrame
    scores_rf = pd.DataFrame({'MSE': MSE_mlr, 'MAPE': MAPE_mlr, 'sMAPE': sMAPE_mlr})

    # Return average accuracy score value of k samples
    return scores_rf.mean()

#----------------------------------------------------------------------------------------------
# Fit GradientBoostingRegressor to all samples of Y1, Y2, ..... Yt
#----------------------------------------------------------------------------------------------

def fit_gbr(parameters, samples_list, ML_model):
    # Define arrays of accuracy scores
    MSE_mlr   = np.empty(parameters[2])
    MAPE_mlr  = np.empty(parameters[2])
    sMAPE_mlr = np.empty(parameters[2])

    for i in range(0, parameters[2]):
        samples_list[i] = pd.DataFrame(samples_list[i], columns = ['X1' , 'X2', 'X3', 'Y'])
        X = samples_list[i].drop('Y', axis = 1) # here we have 3 variables for multiple regression.
        Y = samples_list[i]['Y']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        ML_model.fit(X_train, y_train) #training the algorithm
        y_pred = ML_model.predict(X_test)

        MSE_mlr[i], MAPE_mlr[i], sMAPE_mlr[i] = calculate_perform_metrics(y_test, y_pred)

    # Record MSE, MAPE, sMAPE values of each sample in a DataFrame
    scores_rf = pd.DataFrame({'MSE': MSE_mlr, 'MAPE': MAPE_mlr, 'sMAPE': sMAPE_mlr})

    # Return average accuracy score value of k samples
    return scores_rf.mean()

#---------------------------------------------------------------------------------------------

# Creates a Figure that represents MSE score on y-axis and Year on x-y_axis
def create_plot_MSE(population_scores_mlr, population_scores_rfr, population_scores_gbr, name):
    # Create 3x1 sub plots
    gs = gridspec.GridSpec(3, 1)
    pl.figure(name)

    ax = pl.subplot(gs[0, 0]) # row 0, col 0
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    pl.plot([1,2,3,4,5],[population_scores_gbr['year1'][0], population_scores_gbr['year2'][0],population_scores_gbr['year3'][0],population_scores_gbr['year4'][0], population_scores_gbr['year5'][0]])
    pl.title('GradientBoostingRegressor', fontsize = 12)
    pl.ylabel('MSE', fontsize = 9)
    pl.ylim(0, 3)

    ax = pl.subplot(gs[1, 0]) # row 1, col 0
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    pl.plot([1,2,3,4,5],[population_scores_rfr['year1'][0],population_scores_rfr['year2'][0],population_scores_rfr['year3'][0],population_scores_rfr['year4'][0], population_scores_rfr['year5'][0]])
    pl.title('RandomForestRegressor', fontsize = 12)
    pl.ylabel('MSE', fontsize = 9)
    pl.ylim(0, 3)

    ax = pl.subplot(gs[2, 0]) # row 2, col 0
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    pl.plot([1,2,3,4,5],[population_scores_mlr['year1'][0],population_scores_mlr['year2'][0],population_scores_mlr['year3'][0],population_scores_mlr['year4'][0], population_scores_mlr['year5'][0]])
    pl.title('LinearRegression', fontsize = 12)
    pl.ylabel('MSE', fontsize = 9)
    pl.xlabel('Year')
    pl.ylim(0, 3)

    pl.tight_layout()
    pl.show()

#---------------------------------------------------------------------------------------------
# Creates a Figure of t histograms that represent the distribution of each year of feature X1

def create_histograms(populations_collection, name):
    gs = gridspec.GridSpec(1, 5)
    pl.rc('font', size = 6) # font of x and y axes
    pl.figure(name)
    
    for p in range(0,5):
        year_key = 'year' + str(p+1)
        population = populations_collection[year_key]
        ax = pl.subplot(gs[0, p]) # row 0, col 0
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        pl.title('Year'+str(p+1), fontsize = 12)
        pl.ylabel('Frequency')
        pl.hist(population['X1'], color = 'grey', edgecolor = 'black', alpha=.3)
    pl.tight_layout() # add space between subplots
    pl.show()
