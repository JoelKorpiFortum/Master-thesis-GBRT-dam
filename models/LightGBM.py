# LightGBM.py

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from dateutil.relativedelta import relativedelta
import time
import pickle
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from utils.data_preparation import preprocess_data, split_data, mapping
from processing.custom_metrics import willmotts_d, nash_sutcliffe
from tqdm import tqdm  # progress bar


def process_data_for_target(target, poly_degree=4, test_size=0.3):
    """
    Process data for a given target variable.

    Parameters:
    - target: The target variable to predict.
    - poly_degree: The degree of polynomial features to add (default is 4).
    - test_size: Proportion of data to use as test set (default is 0.3).

    Returns:
    - X_train, X_test, y_train, y_test, split_index, dates, total_months
    """

    features = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18', 'h', \
                'h_MA_007', 'h_MA_014', 'h_RC_007', 'h_RC_030', 'P', 'P_RS_030', \
                'P_RS_060', 'P_RS_090', 'P_RS_180', 'T', 'T_MA_001', 'T_MA_007', \
                't', 'ln_t', 'Cos_s', 'Sin_s', 'Cos_2s', 'Sin_2s', 'month', 'year']
    features.remove(target)
    path = f'./data/LOS_DAMM_{mapping(target)}.csv'
    data = pd.read_csv(path, sep=';', parse_dates=['Date-Time'])

    # Extract start and end dates
    dates = data['Date-Time']
    start_date = dates.iloc[0].date()  # First date (YYYY-MM-DD)
    end_date = dates.iloc[-1].date()  # Last date (YYYY-MM-DD)

    # Preprocess the data (You would need to define preprocess_data elsewhere in your code)
    X, y, dates = preprocess_data(features, target, start_date, end_date, poly_degree=poly_degree, test_size=test_size)

    # Split the data into training and testing sets (Assumes split_data is defined elsewhere)
    X_train, X_test, y_train, y_test, split_index = split_data(X, y, test_size=test_size)

    # Concatenate training and test data for full data (if needed)
    X_all = pd.concat([X_train, X_test])

    # Calculate the separation date between training and test data
    separation_date_train = dates.iloc[split_index].date()
    difference = relativedelta(separation_date_train, start_date)
    total_months = difference.years * 12 + difference.months

    # Return the processed results
    return X, y, X_train, X_test, y_train, y_test, X_all, split_index, dates, total_months


# Helper function to evaluate 3-fold time series cross validation
def timecv_model(model, X, y):
    tfold = TimeSeriesSplit(n_splits=3)
    rmse_list = []  # To store RMSE for each fold
    for _, (train_index, val_index) in tqdm(enumerate(tfold.split(X), start=1)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        reg = model.fit(X_train, y_train)
        pred_val = reg.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred_val))
        rmse_list.append(rmse)
    return rmse_list


def cv_result(model, X, y):
    model_name = model.__class__.__name__
    rmse_ = timecv_model(model, X, y)
    for i, rmse in enumerate(rmse_):
        print(f'{i+1}th fold: {model_name} RMSE: {rmse:.4f}')
    avg_rmse = np.mean(rmse_)
    print(f'\n{model_name} average RMSE: {avg_rmse:.4f}')
    return avg_rmse


def objective_lgb(trial, X_train, y_train):
    """
    The objective function to tune hyperparameters. It evaluate the score on a
    validation set. This function is used by Optuna, a Bayesian tuning framework.
    """

    params = {
            'objective': 'regression',
            'verbose': -1,
            'random_state': 42,
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 2, 30),
            'learning_rate': trial.suggest_float("learning_rate", 1e-5, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 1),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),               # L1 regularization term
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),             # L2 regularization term (similar to your 'l2_regularization')
            # 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'linear_tree': trial.suggest_categorical('linear_tree', [True, False])
            }
    model = LGBMRegressor(**params)
    avg_rmse = cv_result(model, X_train, y_train)
    return avg_rmse


def lgb_tune(target, X_train, y_train, n_trials):
    # Uncomment to suppress optuna log messages
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42)
    study_model = optuna.create_study(direction = 'minimize', sampler = sampler, study_name=f'hyperparameters_tuning_{target}')
    study_model.optimize(lambda trial: objective_lgb(trial, X_train, y_train), n_trials = n_trials)  # type: ignore

    trial = study_model.best_trial
    best_params = trial.params
    print('Best params from optuna: \n', best_params)
    return best_params


def lgb_predict_evaluate(best_params, X_train, X_test, y_train, y_test, X_all):
    # Fit model
    opt_model = LGBMRegressor(**best_params, verbose=-1)
    opt_model.fit(X_train, y_train)

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
    return all_predictions, rmse_train, rmse_test, mae_test, d_test, NSE_test


# def perm_feat(X, opt_model):
#   # Compute permutation importance
#   actual_feature_names = X.columns
#   perm = permutation_importance(opt_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
#   perm_idx = perm.importances_mean.argsort()  # type: ignore
#   perm_importances = perm.importances[perm_idx].T  # type: ignore
#   perm_labels = actual_feature_names[perm_idx]

#   fea_imp_ = pd.DataFrame({'cols':X_train.columns, 'fea_imp':opt_model.feature_importances_})  # type: ignore
#   fea_labels = fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False)
#   return 0


if __name__ == '__main__':
    start_time = time.time()
    Response_variables = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18']
    output_lines = []  # List to store output for each target
    for target in Response_variables:
        start_trial_time = time.time()
        output_lines.append(f"TARGET: {target}\n")  # Add target header
        poly_degree = 4
        test_size = 0.3
        n_trials = 2
        X, y, X_train, X_test, y_train, y_test, X_all, split_index, dates, total_months = process_data_for_target(target=target, poly_degree=poly_degree, test_size=test_size)
        best_params = lgb_tune(target, X_train, y_train, n_trials)
        all_predictions, rmse_train, rmse_test, mae_test, d_test, NSE_test = lgb_predict_evaluate(best_params, X_train, X_test, y_train, y_test, X_all)
        
        plotting_data = {
            'X': X,
            'actual_y': y,
            'X_all': X_all,
            'dates': dates,
            'predictions': all_predictions,
            'split_index': split_index,
            'RMSE': rmse_test,
            'MAE': mae_test,
            'WILMOTT': d_test,
            'NSE': NSE_test,
            # 'perm_importances': perm_importances,
            # 'perm_labels':perm_labels,
            # 'native_importance': fea_imp_,
            # 'native_labels': fea_labels
        }

        # Pickle: save the plotting data and model to serial files
        with open(f'./visualization/plotting_data/LightGBM/LightGBM_{target}_plotting_data.pkl', 'wb') as f:
            pickle.dump(plotting_data, f)

        opt_model = LGBMRegressor(**best_params, verbose=-1)

        with open(f'./visualization/models/LightGBM/LightGBM_model_{target}.pkl', 'wb') as file:
            pickle.dump(opt_model, file)

        print("\n~~~ TARGET ~~~")
        print(target)
        print("\n~~~ MODEL ~~~")
        print(f"Best parameters: {best_params}")
        print("\n~~~ TEST METRICS ~~~")
        print(f"RMSE_train: {rmse_train}")
        print(f"RMSE_test: {rmse_test}")
        print(f"MAE_test: {mae_test}")
        print(f"Willmott's d Test: {d_test}")
        print(f"Nash-Sutcliffe Test: {NSE_test}")
        print("\n~~~ OTHER STATS ~~~")
        print(f"Train data length: {total_months} months")

        # Append each piece of information to the output list
        output_lines.append(f"Best parameters: {best_params}\n")
        output_lines.append(f"RMSE_train: {rmse_train}\n")
        output_lines.append(f"RMSE_test: {rmse_test}\n")
        output_lines.append(f"MAE_test: {mae_test}\n")
        output_lines.append(f"Willmott's d Test: {d_test}\n")
        output_lines.append(f"Nash-Sutcliffe Test: {NSE_test}\n")
        output_lines.append(f"Train data length: {total_months} months\n")

        end_trial_time = time.time()
        trial_time = end_trial_time - start_trial_time
        print(f"Trial time: {trial_time:.4f} seconds\n")
        output_lines.append(f"Trial time: {trial_time:.4f} seconds\n")
        output_lines.append("\n")  # Add an empty line for separation
    
    final_time = time.time()
    elapsed_time = final_time - start_time
    print(f"Total elapsed time: {elapsed_time:.4f} seconds\n")
    output_lines.append(f"Total elapsed time: {elapsed_time:.4f} seconds\n")
    output_lines.append("\n")  # Add an empty line for separation
    # Write all metrics to a text file
    with open("LightGBM_output.txt", "w") as file:
        file.writelines(output_lines)

    print("Output saved to LightGBM_output.txt")