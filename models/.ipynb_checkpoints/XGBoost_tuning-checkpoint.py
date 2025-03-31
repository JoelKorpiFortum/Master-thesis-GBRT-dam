# XGBoost.py

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
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
from processing.custom_metrics import nash_sutcliffe, kling_gupta
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

    features = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18',
                'h', 'h_MA_007', 'h_MA_014', 'h_MA_030', 'h_MA_060', 'h_RC_007', 'h_RC_030',
                'P', 'P_RS_030', 'P_RS_060', 'P_RS_090', 'P_RS_180',
                'T', 'T_MA_001', 'T_MA_007', 'T_MA_030', 'T_MA_060',
                't', 'ln_t', 'Cos_s', 'Sin_s', 'Cos_2s', 'Sin_2s', 'month', 'year']
    if target in features:
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


def cv_result(model, X, y, penalty_factor=0.3):
    model_name = model.__class__.__name__
    rmse_list = timecv_model(model, X, y)

    for i, rmse in enumerate(rmse_list):
        print(f'{i+1}th fold: {model_name} RMSE: {rmse:.4f}')

    avg_rmse = np.mean(rmse_list)
    max_rmse = np.max(rmse_list)

    print(f'\n{model_name} average RMSE: {avg_rmse:.4f}')
    print(f'{model_name} worst RMSE: {max_rmse:.4f}')

    # Directly combine the average and maximum RMSE
    penalty = penalty_factor * (max_rmse - avg_rmse)
    total_loss = avg_rmse + penalty
    print(f'Corresponding penalty value: {total_loss:.4f}')

    return total_loss, avg_rmse, rmse_list


def objective_xgb(trial, X_train, y_train, penalty_factor):
    """
    The objective function to tune hyperparameters. It evaluate the score on a
    validation set. This function is used by Optuna, a Bayesian tuning framework.
    """

    params = {
            # 'eval_metric': 'rmse',
            'random_state': 42,
            'tree_method': 'hist',
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'learning_rate': trial.suggest_float("learning_rate", 1e-5, 0.09901, step=0.001),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'max_leaves': trial.suggest_int('max_leaves', 2, 30),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1, step=0.01),
            'subsample': trial.suggest_float('subsample', 0.5, 1, step=0.01),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, step=0.01),               # L1 regularization term
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10, step=0.01),             # L2 regularization term (similar to your 'l2_regularization')
            'gamma': trial.suggest_float('gamma', 0, 5, step=0.01),
            }

    model = XGBRegressor(**params)
    total_loss, avg_rmse, _ = cv_result(model, X_train, y_train, penalty_factor)
    del model
    return total_loss


def xgb_tune(target, X_train, y_train, n_trials, penalty_factor):
    # Uncomment to suppress optuna log messages
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42)
    study_model = optuna.create_study(direction = 'minimize', sampler = sampler, study_name=f'hyperparameters_tuning_{target}')
    study_model.optimize(lambda trial: objective_xgb(trial, X_train, y_train, penalty_factor), n_trials = n_trials)  # type: ignore

    trial = study_model.best_trial
    best_params = trial.params
    print('Best params from optuna: \n', best_params)
    del study_model
    return best_params

def xgb_predict_evaluate(best_params, X_train, X_test, y_train, y_test, X_all, penalty_factor):
    # Fit model
    opt_model = XGBRegressor(**best_params)
    opt_model.fit(X_train, y_train)

    # Predictions
    train_predictions = opt_model.predict(X_train)
    test_predictions = opt_model.predict(X_test)
    all_predictions = opt_model.predict(X_all)

    # Evaluation
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
    rmse_crossval = cv_result(opt_model, X_train, y_train, penalty_factor)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    mae_test = mean_absolute_error(y_test, test_predictions)
    NSE_test = nash_sutcliffe(y_test, test_predictions)
    KGE_test = kling_gupta(y_test, test_predictions)
    return all_predictions, rmse_train, rmse_crossval, rmse_test, mae_test, NSE_test, KGE_test
    
# # Compute train/test scores
# train_score = np.zeros((n_splits), dtype=np.float64)   
# for i, y_pred in enumerate(model.staged_predict(X_train)):
#     train_score[i] = mean_squared_error(y_train, y_pred)

# test_score = np.zeros((n_splits), dtype=np.float64)
# for i, y_pred in enumerate(model.staged_predict(X_test)):
#     test_score[i] = mean_squared_error(y_test, y_pred)
# iterations = np.arange(n_splits) + 1

# # Compute permutation importance
# actual_feature_names = X.columns
# perm = permutation_importance(opt_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
# perm_idx = perm.importances_mean.argsort() #type: ignore
# perm_importances = perm.importances[perm_idx].T #type: ignore
# perm_labels = actual_feature_names[perm_idx]

if __name__ == '__main__':
    start_time = time.time()
    Response_variables = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18']
    results = []  # List to store output for each target

    poly_degree = 4
    test_size = 0.3
    n_trials = 50
    penalty_factor = 0.1

    for target in Response_variables:
        start_trial_time = time.time()

        X, y, X_train, X_test, y_train, y_test, X_all, split_index, dates, total_months = process_data_for_target(target=target, poly_degree=poly_degree, test_size=test_size)
        best_params = xgb_tune(target, X_train, y_train, n_trials, penalty_factor)
        all_predictions, rmse_train, rmse_crossval, rmse_test, mae_test, NSE_test, KGE_test  = xgb_predict_evaluate(best_params, X_train, X_test, y_train, y_test, X_all, penalty_factor)

        rmse_mean_crossval = rmse_crossval[1]
        rmse_crossval_folds = rmse_crossval[2]

        plotting_data = {
            'X': X,
            'actual_y': y,
            'X_all': X_all,
            'dates': dates,
            'predictions': all_predictions,
            'split_index': split_index,
            'RMSE_crossval': rmse_mean_crossval,
            'RMSE': rmse_test,
            'MAE': mae_test,
            'NSE': NSE_test,
            'KGE': KGE_test,
            # 'perm_importances': perm_importances,
            # 'perm_labels':perm_labels,
            # 'native_importance': fea_imp_,
            # 'native_labels': fea_labels
        }

        # Pickle: save the plotting data and model to serial files
        with open(f'./visualization/plotting_data/XGBoost/XGBoost_{target}_plotting_data.pkl', 'wb') as f:
            pickle.dump(plotting_data, f)

        opt_model = XGBRegressor(**best_params)

        with open(f'./visualization/models/XGBoost/XGBoost_model_{target}.pkl', 'wb') as file:
            pickle.dump(opt_model, file)

        print("\n~~~ TARGET ~~~")
        print(target)
        print("\n~~~ MODEL ~~~")
        print(f"Best parameters: {best_params}")
        print("\n~~~ TEST METRICS ~~~")
        print(f"RMSE_train: {rmse_train:.3f}")
        print(f"RMSE_crossval: {rmse_mean_crossval:.3f}")
        print(f"RMSE_test: {rmse_test:.3f}")
        print(f"MAE_test: {mae_test:.3f}")
        print(f"Nash-Sutcliffe Test: {NSE_test:.3f}")
        print(f"Kling-Gupta Test: {KGE_test:.3f}")
        print("\n~~~ OTHER STATS ~~~")
        print(f"Train data length: {total_months} months")
        print(f"penalty_factor: {penalty_factor}")

        end_trial_time = time.time()
        trial_time = end_trial_time - start_trial_time
        print(f"Trial time: {trial_time:.4f} seconds\n")

        results.append({
            'Target': target,
            'Best Parameters': str(best_params),
            'RMSE_train': np.round(rmse_train, 3),
            'RMSE_crossval': np.round(rmse_mean_crossval, 3),
            'RMSE_test': np.round(rmse_test, 3),
            'MAE_test': np.round(mae_test, 3),
            'Nash-Sutcliffe Test': np.round(NSE_test, 3),
            'Kling-Gupta Test': np.round(KGE_test, 3),
            'Train data length (months)': total_months,
            'Penalty factor': penalty_factor,
            'Number of trials': n_trials,
            'Trial time (seconds)': np.round(trial_time, 4)
        })

    final_time = time.time()
    elapsed_time = final_time - start_time
    print(f"Total elapsed time: {elapsed_time:.4f} seconds\n")

    df_results = pd.DataFrame(results)
    # df_results.to_csv("LightGBM_output_results.csv", index=False)

    # Write all metrics to a text file
    with open(f"./tuning_output/XGBoost/XGBoost_tuning_31_3_pen_{penalty_factor*10:02.0f}.txt", "w") as file:
        file.writelines(df_results.to_string(index=False))
        file.write(f"\n\nTotal elapsed time: {elapsed_time:.4f} seconds")

    print(f"Output saved to XGBoost_tuning_31_3_pen_{penalty_factor*10:02.0f}.txt")