# GBRT.py

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from dateutil.relativedelta import relativedelta
import time
import pickle
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from utils.data_preparation import preprocess_data, split_data, mapping
from processing.custom_metrics import willmotts_d, nash_sutcliffe
import multiprocessing
from scipy.stats import pearsonr
from tqdm import tqdm  # progress bar

start_time = time.time()

# Response_variables = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18']

features = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18', 'h', \
            'h_MA_007', 'h_MA_014', 'h_RC_007', 'h_RC_030', 'P', 'P_RS_030', \
            'P_RS_060', 'P_RS_090', 'P_RS_180', 'T', 'T_MA_001', 'T_MA_007', \
            't', 'ln_t', 'Cos_s', 'Sin_s', 'Cos_2s', 'Sin_2s', 'month', 'year']
target = 'GV1'
features.remove(target)

poly_degree = 4
path = f'./data/LOS_DAMM_{mapping(target)}.csv'
data = pd.read_csv(path, sep=';', parse_dates=['Date-Time'])

dates = data['Date-Time']
start_date = dates.iloc[0].date()  # First date (YYYY-MM-DD)
end_date = dates.iloc[-1].date()  # Last date (YYYY-MM-DD)

test_size = 0.3
split_index = int(len(data) * (1 - test_size))
separation_date = dates.iloc[split_index].date()
difference = relativedelta(separation_date, start_date)
total_months = difference.years * 12 + difference.months

X, y, dates = preprocess_data(features, target, start_date, end_date, poly_degree=poly_degree, test_size=test_size)
X_train, X_test, y_train, y_test, split_index = split_data(X, y, test_size=test_size)
X_all = pd.concat([X_train, X_test])

# tscv = TimeSeriesSplit(n_splits=3, test_size=int(0.1*len(X)))

# 3-fold time series cross validation
def timecv_model(model, X, y):
    tfold = TimeSeriesSplit(n_splits=3, test_size=int(0.1*len(X)))
    pcc_list = []
    for _, (train_index, test_index) in tqdm(enumerate(tfold.split(X), start=1)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        reg = model.fit(X_train, y_train)
        pred = reg.predict(X_test)
        pcc = pearsonr(pred, y_test) 
        pcc_list.append(pcc[0])
    
    return pcc_list

def cv_result(model, X, y):
    model_name = model.__class__.__name__
    pcc_ = timecv_model(model, X, y)
    for i, pcc in enumerate(pcc_):
        print(f'{i}th fold: {model_name} PCC: {pcc:.4f}')
    print(f'\n{model_name} average PCC: {np.mean(pcc_):.4f}')

# Uncomment to suppress optuna log messages
# optuna.logging.set_verbosity(optuna.logging.WARNING)

sampler = TPESampler(seed=42)

def objective_gbrt(trial):
    """
    The objective function to tune hyperparameters. It evaluate the score on a
    validation set. This function is used by Optuna, a Bayesian hyperparameter tuning framework.

    """

    params = {
            'verbose': 0,
            'random_state': 42,
            'max_iter': trial.suggest_int('max_iter', 500, 3000),
            'learning_rate': trial.suggest_float("learning_rate", 1e-5, 0.1),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 30),
            'l2_regularization': trial.suggest_float('l2_regularization', 0, 5),
            'max_features': trial.suggest_float('max_features', 0.5, 1),
            'early_stopping': trial.suggest_categorical('early_stopping', [True, 'auto'])
            }
    
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)  # type: ignore
    pred = model.predict(X_test) 
    rmse = np.sqrt(mean_squared_error(pred, y_test))
    # pcc = pearsonr(pred, y_test)[0]
    return rmse

study_model = optuna.create_study(direction = 'minimize', sampler = sampler, study_name='hyperparameters_tuning')
study_model.optimize(objective_gbrt, n_trials = 30)  # type: ignore

trial = study_model.best_trial
best_params = trial.params

print('Best params from optuna: \n', best_params)

opt_model = HistGradientBoostingRegressor(**best_params)

cv_result(opt_model, X, y)

# Predictions
train_predictions = opt_model.predict(X_train)
test_predictions = opt_model.predict(X_test)
all_predictions = opt_model.predict(X_all)

# Evaluation
rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
mae_test = mean_absolute_error(y_test, test_predictions)
d_test = willmotts_d(y_test, test_predictions)
NSE_test = nash_sutcliffe(y_test, test_predictions)

# # Compute train/test scores
# train_score = -opt_model.train_score_
# test_score = np.zeros((opt_model.n_iter_,), dtype=np.float64)
# for i, y_pred in enumerate(opt_model.staged_predict(X_test)):
#     test_score[i] = mean_squared_error(y_test, y_pred)
# iterations = np.arange(opt_model.n_iter_) + 1

# Compute permutation importance
actual_feature_names = X.columns
perm = permutation_importance(opt_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
perm_idx = perm.importances_mean.argsort() #type: ignore
perm_importances = perm.importances[perm_idx].T #type: ignore
perm_labels = actual_feature_names[perm_idx]

# Prepare data for plotting
plotting_data = {
    'X': X,
    'X_all': X_all,
    'dates': dates,
    'actual_y': y,
    'predictions': all_predictions,
    'split_idx': split_index,
    'RMSE': rmse_test,
    'MAE': mae_test,
    'WILMOTT': d_test,
    'NSE': NSE_test,
    # 'train_score': train_score,
    # 'test_score': test_score,
    # 'iterations': iterations,
    'perm_importances': perm_importances,
    'perm_labels': perm_labels
}

model_type = 'GBRT'

# Pickle: save the plotting data and model to serial files
with open(f'./visualization/plotting_data/{model_type}/{model_type}_{target}_plotting_data.pkl', 'wb') as f:
    pickle.dump(plotting_data, f)

with open(f'./visualization/models/{model_type}/{model_type}_model_{target}.pkl', 'wb') as file:
    pickle.dump(opt_model, file)

end_time = time.time()
elapsed_time = end_time - start_time

feature_names = X_train.columns
print("Features used in the GBRT model:")
print(feature_names)

print(f"\n~~~ MODEL ~~~ \n")
print(f"Best parameters: {best_params}")
print(f"\n~~~ TEST METRICS ~~~ \n")
# print(f"RMSE_CV: {rmse_cv}")
print(f"RMSE_train: {rmse_train}")
print(f"RMSE_test: {rmse_test}")
print(f"MAE_test: {mae_test}")
print(f"Willmott's d Test: {d_test}")
print(f"Nash-Sutcliffe Test: {NSE_test}")
print(f"\n~~~ OTHER STATS ~~~ \n")
print(f"Train data length: {total_months} months")
# print(f"N iter: {model.n_iter_}")
print(f"Elapsed time: {elapsed_time:.4f} seconds\n")

# param_grid = {
#     'max_iter': [500],     # Boosting rounds
#     'learning_rate': [0.01],  # eta
#     'max_depth': [3],
#     'min_samples_leaf': [10],
#     'l2_regularization': [1],
#     'early_stopping': ["auto"],
#     'random_state': [42]
# }

# grid_search = GridSearchCV(
#     HistGradientBoostingRegressor(), 
#     param_grid, 
#     cv=tscv, 
#     scoring='neg_root_mean_squared_error', 
#     n_jobs=-1)

# grid_search.fit(X_train, y_train)
# best_params = grid_search.best_params_
# model = grid_search.best_estimator_ 

# cv_results = grid_search.cv_results_
# best_index = grid_search.best_index_
# rmse_cv = -cv_results['mean_test_score'][best_index]