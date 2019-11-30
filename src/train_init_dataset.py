# Import libraries
import pandas as pd
# Import helping functions
from  functions import draw_k_samples, fit_lr, fit_rfr, fit_gbr
from main import parameters, my_models


df = pd.read_csv('dataset-generator/data/init_dataset.csv', sep = ",")

# Draw k samples from df
samples_list = draw_k_samples(parameters, df)

# Next we want to fit chosen models on each sample of t populations
# Linear Regression, Random Forest Regressor, Gradient Boosting Regressor

scores_mlr =  fit_lr(parameters, samples_list, my_models[0])
scores_rfr =  fit_rfr(parameters, samples_list, my_models[1])
scores_gbr =  fit_gbr(parameters, samples_list, my_models[2])

print('Training of artificially created dataset is finished')
