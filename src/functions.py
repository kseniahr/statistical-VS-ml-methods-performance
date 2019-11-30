import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
# Import visualization libraries
import numpy as np


# Call np.random.seed() to make examples with (pseudo) random data reproducible:
np.random.seed(1234)


def build_regression_dataset():
    ''' A possible way to generate a random regression problem '''
    # Generate a random regression problem
    X, y = make_regression(n_samples=1000, n_features=3, n_informative=2,
                           n_targets=1, random_state=100, noise=0.05)
    #y *= 0.01

    return pd.DataFrame({'X1': X[:,0], 'X2': X[:,1], 'X3': X[:,2], 'Y': y})

#---------------------------------------------------------------------------------------------

def change_vars_distribution(parameters, coefficients, populations_collection):
    '''np.random.normal draws random numbers from a normal distribution'''
    # the arguments of np.random.normalnorm(mean, sd, number of random numbers N)
    # the scale function transforms the input values in a range from 0 to 1
    # I used a autoregressive function, meaning that t1 will influence the values on t2, t3 will influence t4 and so on
    X1 = scale(populations_collection['X1'] + np.random.normal(0.0, 1.0, parameters[0]))
    X2 = scale(populations_collection['X2'] + np.random.normal(0.0, 1.0, parameters[0]))
    X3 = scale(populations_collection['X3'] + np.random.normal(0.0, 1.0, parameters[0]))

    error = np.random.normal(0.0, coefficients[4], parameters[0])

    # calculate endogene variables
    Y = coefficients[0] + coefficients[1]*X1 + coefficients[2]*X2 + coefficients[3]*X3 + error

    independent_vars = [X1, X2, X3]

    return independent_vars, error, Y

#---------------------------------------------------------------------------------------------

# Generate exogene variables (including latent variables like prediction-error)
def generate_variables(parameters, coefficients):
    # seed is used to keep the same X1..Xn variables for each population
    np.random.seed(42)
    X1 = np.random.normal(0.0, 1.0, parameters[0])
    np.random.seed(43)
    X2 = np.random.normal(0.0, 1.0, parameters[0])
    np.random.seed(44)
    X3 = np.random.normal(0.0, 1.0, parameters[0])

    error = np.random.normal(0.0, coefficients[4], parameters[0])

    # calculate endogene variables
    Y = coefficients[0] + coefficients[1]*X1 + coefficients[2]*X2 + coefficients[3]*X3 + error

    independent_variables = [X1, X2, X3]

    return independent_variables, error, Y

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
# check median aggregation on simulation studies
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


#
# f = plt.figure(figsize=(19, 15))
# plt.matshow(df.corr(), fignum=f.number)
# plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
# plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16);
