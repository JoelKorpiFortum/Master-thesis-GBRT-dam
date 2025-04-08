# GBRT.py

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
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
from models.best_params.GBRT_saved_params import HYPERPARAMETERS
import multiprocessing

N_CORES = multiprocessing.cpu_count()
# For dual process parallelization
HALF_CORES = max(1, N_CORES // 2)
N_JOBS = max(1, N_CORES - 1)


def process_data_for_target(target, poly_degree=0, test_size=0.1):
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
                'P', 'P_RS_007', 'P_RS_030', 'P_RS_060', 'P_RS_090', 'P_RS_180',
                'TA', 'TA_MA_001', 'TA_MA_007', 'TA_MA_030', 'TA_MA_060',
                'TW', 'TW_MA_007', 'TW_MA_030',
                'Q', 'Q_RS_007', 'Q_RS_030',
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


def tscv_model(model, X, y):
    """ Evaluate 3-fold time series cross validation """
    n_splits = 3
    tfold = TimeSeriesSplit(n_splits=n_splits, test_size=int(0.1*len(X)))
    cv_rmse = np.full(n_splits, np.nan)  # To store RMSE for each fold
    for i, (train_index, val_index) in tqdm(enumerate(tfold.split(X))):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        fit_model = model.fit(X_train, y_train)
        y_val_pred = fit_model.predict(X_val)
        cv_rmse[i] = np.sqrt(mean_squared_error(y_val, y_val_pred))
        print(f'\tRMSE: {cv_rmse[i]:.4f}')
    return cv_rmse


def objective_gbrt(trial, X_train, y_train):
    """
    The objective function to tune hyperparameters. It evaluate the score on a
    validation set. This function is used by Optuna, a Bayesian tuning framework.
    """

    params = {
            # 'eval_metric': 'rmse',
            'random_state': 42,
            'max_iter': trial.suggest_int('max_iter', 1000, 3000, log=True),
            'learning_rate': trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 12),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 26),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 100, log=True),
            'max_features': trial.suggest_float('max_features', 0.1, 1, log=True),
            'early_stopping': trial.suggest_categorical('early_stopping', [True, 'auto'])
            }
    
    model = HistGradientBoostingRegressor(verbose=0, **params)
    cv_rmse = tscv_model(model, X_train, y_train)
    del model
    return np.mean(cv_rmse)


def gbrt_tune(target, X_train, y_train, n_trials):
    # Uncomment to suppress optuna log messages
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42)
    study_model = optuna.create_study(direction='minimize', sampler=sampler, study_name=f'hyperparameter_tuning_{target}')
    study_model.optimize(lambda trial: objective_gbrt(trial, X_train, y_train), n_trials=n_trials)  # type: ignore

    trial = study_model.best_trial
    best_params = trial.params
    print('Best params from optuna: \n', best_params)
    del study_model
    return best_params


def get_best_params(target):
    with open(f'./models/best_params/GBRT_best_params.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)
    return best_params_dict.get(target, None)  # Returns the hyperparameters for the target

def get_saved_params(target):
    return HYPERPARAMETERS.get(target, None)  # Returns the hyperparameters for the target


def gbrt_predict_evaluate(best_params, X_train, X_test, y_train, y_test, X_all):
    # Fit model
    opt_model = HistGradientBoostingRegressor(verbose=0, random_state=42, **best_params)
    opt_model.fit(X_train, y_train)

    # Predictions
    train_predictions = opt_model.predict(X_train)
    test_predictions = opt_model.predict(X_test)
    all_predictions = opt_model.predict(X_all)

    # Evaluation
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
    rmse_crossval_folds = tscv_model(opt_model, X_train, y_train)
    rmse_crossval = np.mean(rmse_crossval_folds)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    mae_test = mean_absolute_error(y_test, test_predictions)
    NSE_test = nash_sutcliffe(y_test, test_predictions)
    KGE_test = kling_gupta(y_test, test_predictions)
    return opt_model, all_predictions, rmse_train, rmse_crossval, rmse_test, mae_test, NSE_test, KGE_test

# def sklearn_hist_gbt_feature_importances(sklearn_hist_gbt):
#     # sum up metric gain for each feature over all of the trees

#     # `feature_idx` is *not* the index of the feature in `feature_names_in_`!!!
#     feature_idx_to_feature_name = {}
#     i = 0
#     for indices in reg._preprocessor._transformer_to_input_indices.values():
#         for j in indices:
#             feature_idx_to_feature_name[i] = reg.feature_names_in_[j]
#             i += 1

#     dfs = []
#     for tree in sklearn_hist_gbt._predictors:
#         assert len(tree) == 1
#         tree = tree[0]
#         df = pd.DataFrame(tree.nodes)
#         dfs.append(df)

#     df = pd.concat(dfs)
#     # see https://github.com/scikit-learn/scikit-learn/blob/d74a5a5c4c427c81292b15b92df8138e70fd94b9/sklearn/ensemble/_hist_gradient_boosting/grower.py#L729
#     # need to be careful to not sum the "gain" contribution from leaf nodes.
#     # these array of nodes are all initialized as zeros, so a "feature_idx" of zero might refer to the first feature or (for leaf nodes) it might just be the default.
#     # The metric gain in the leaf nodes is a "potential metric gain" if a further split was performed!!
#     branch_nodes = df.loc[df["is_leaf"] == 0]
#     branch_nodes["feature_name"] = branch_nodes["feature_idx"].map(feature_idx_to_feature_name)
#     return branch_nodes.groupby("feature_name")["gain"].sum().sort_values(ascending=False).to_dict()


# def feature_prune_rfecv(best_params, X_train, X_test, y_train, y_test, rmse_train, rmse_test, mae_test):
#     """
#     Performs recursive feature elimination to select optimal features
#     and compares pruned model performance with the full-feature model.

#     Parameters:
#     -----------
#     best_params : dict
#         Parameters for HistGradientBoostingRegressor
#     X_train, X_test : DataFrame
#         Training and test feature sets
#     y_train, y_test : Series
#         Training and test target values
#     rmse_train, rmse_test, mae_test : float, optional
#         Metrics from the full-feature model for comparison

#     Returns:
#     --------
#     tuple
#         (Pruned model, number of optimal features, performance comparison dict)
#     """

#     # Create and fit a basic model first
#     base_model = HistGradientBoostingRegressor(**best_params, verbose=0)
#     base_model.fit(X_train, y_train)

#     # Use permutation importance to get feature rankings
#     perm = permutation_importance(
#         base_model, X_train, y_train, 
#         n_repeats=10,
#         random_state=42,
#         n_jobs=-1
#     )

#     # Sort features by importance
#     importances = perm.importances_mean
#     indices = np.argsort(importances)[::-1]  # Descending order

#     # Set up cross-validation
#     n_splits = 3
#     cv = TimeSeriesSplit(n_splits=n_splits, test_size=int(0.1*len(X_train)))

#     # Try different numbers of features and evaluate performance
#     n_features_to_try = range(5, len(X_train.columns) + 1, 5)  # Try in steps of 5
#     cv_scores = []
#     feature_sets = []

#     for n_features in n_features_to_try:
#         # Select top n_features
#         top_indices = indices[:n_features]
#         selected_features = X_train.columns[top_indices]
#         feature_sets.append(selected_features)

#         # Evaluate with cross-validation
#         model = HistGradientBoostingRegressor(**best_params, verbose=0)
#         scores = cross_val_score(
#             model, 
#             X_train[selected_features], 
#             y_train,
#             cv=cv,
#             scoring='neg_mean_squared_error',
#             n_jobs=-1
#         )
#         cv_scores.append(-np.mean(scores))  # Convert to positive RMSE

#     # Find optimal number of features
#     optimal_idx = np.argmin(cv_scores)
#     optimal_features = len(feature_sets[optimal_idx])
#     best_features = feature_sets[optimal_idx]

#     print(f"Optimal number of features: {optimal_features}")
#     print(f"Selected features: {best_features.tolist()}")

#     # Create feature ranking similar to RFECV output
#     all_features = X_train.columns.tolist()
#     rankings = [1 if feat in best_features else 2 for feat in all_features]

#     feature_ranking = pd.DataFrame({
#         'Feature': all_features,
#         'Ranking': rankings,
#         'Importance': [importances[list(X_train.columns).index(feat)] for feat in all_features]
#     }).sort_values(by=['Ranking', 'Importance'], ascending=[True, False]).reset_index(drop=True)

#     print(feature_ranking)

#     # Create pruned datasets and train model
#     X_train_pruned = X_train[best_features]
#     X_test_pruned = X_test[best_features]
#     gbrt_prune = HistGradientBoostingRegressor(**best_params, verbose=0)
#     gbrt_prune.fit(X_train_pruned, y_train)

#     train_preds_pruned = gbrt_prune.predict(X_train_pruned)
#     test_preds_pruned = gbrt_prune.predict(X_test_pruned)

#     rmse_train_pruned = np.sqrt(mean_squared_error(y_train, train_preds_pruned))
#     rmse_test_pruned = np.sqrt(mean_squared_error(y_test, test_preds_pruned))
#     mae_test_pruned = mean_absolute_error(y_test, test_preds_pruned)

#     # Comparison metrics
#     comparison = {
#         'rmse_train': {'full': rmse_train, 'pruned': rmse_train_pruned, 'improvement': (rmse_train - rmse_train_pruned) / rmse_train * 100},
#         'rmse_test': {'full': rmse_test, 'pruned': rmse_test_pruned, 'improvement': (rmse_test - rmse_test_pruned) / rmse_test * 100},
#         'mae_test': {'full': mae_test, 'pruned': mae_test_pruned, 'improvement': (mae_test - mae_test_pruned) / mae_test * 100}
#     }

#     print(f"New RMSE train: {rmse_train_pruned:.4f}, Old: {rmse_train:.4f}, Improvement: {comparison['rmse_train']['improvement']:.2f}%")
#     print(f"New RMSE test: {rmse_test_pruned:.4f}, Old: {rmse_test:.4f}, Improvement: {comparison['rmse_test']['improvement']:.2f}%")
#     print(f"New MAE test: {mae_test_pruned:.4f}, Old: {mae_test:.4f}, Improvement: {comparison['mae_test']['improvement']:.2f}%")

#     return gbrt_prune, optimal_features


if __name__ == '__main__':
    start_time = time.time()
    elapsed_time = 0.0
    targets = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18']
    results = []  # List to store output for each target
    params_dict = {}  # Dict to store optimal hyperparameters for each target
    poly_degree = 0  # Exponent for the HST model's 'h' polynomial feature
    test_size = 0.2

    tuning = False  # True = hyperparameter tuning, False = inference
    n_trials = 30
    prune = False  # True = pruning is performed with RFECV

    for target in targets:

        X, y, X_train, X_test, y_train, y_test, X_all, split_index, dates, total_months = process_data_for_target(target=target, poly_degree=poly_degree, test_size=test_size)

        if tuning:
            start_tune_time = time.time()
            best_params = gbrt_tune(target, X_train, y_train, n_trials)
            end_tune_time = time.time()
            tune_time = end_tune_time - start_tune_time
        else:
            # best_params = get_best_params(target)
            best_params = get_saved_params(target)

        start_inference_time = time.time()    
        opt_model, all_predictions, rmse_train, rmse_crossval, rmse_test, mae_test, NSE_test, KGE_test = gbrt_predict_evaluate(best_params, X_train, X_test, y_train, y_test, X_all)
        end_inference_time = time.time()

        inference_time = end_inference_time - start_inference_time
        if tuning:
            print(f"Target tuning time: {tune_time:.4f} seconds\n")
        print(f"Target inference time: {inference_time:.4f} seconds\n")

        # if prune:
        #     gbrt_reg_prune, optimal_features = feature_prune_rfecv(best_params, X_train, X_test, y_train, y_test, rmse_train, rmse_test, mae_test)

        if tuning:
            params_dict[target] = best_params

        plotting_data = {
            'X': X,
            'y': y,
            'X_all': X_all,
            'dates': dates,
            'predictions': all_predictions,
            'split_index': split_index,
            'RMSE_crossval': rmse_crossval,
            'RMSE': rmse_test,
            'MAE': mae_test,
            'NSE': NSE_test,
            'KGE': KGE_test
        }

        # Pickle: save the plotting data and model to serial files
        with open(f'./visualization/plotting_data/GBRT/GBRT_{target}_plotting_data.pkl', 'wb') as f:
            pickle.dump(plotting_data, f)

        opt_model = HistGradientBoostingRegressor(**best_params)  # type: ignore

        with open(f'./visualization/models/GBRT/GBRT_model_{target}.pkl', 'wb') as file:
            pickle.dump(opt_model, file)

        print("\n~~~ TARGET ~~~")
        print(target)
        print("\n~~~ MODEL ~~~")
        print(f"Best parameters: {best_params}")
        print("\n~~~ TEST METRICS ~~~")
        print(f"RMSE_train: {rmse_train:.3f}")
        print(f"RMSE_crossval: {rmse_crossval:.3f}")
        print(f"RMSE_test: {rmse_test:.3f}")
        print(f"MAE_test: {mae_test:.3f}")
        print(f"Nash-Sutcliffe Test: {NSE_test:.3f}")
        print(f"Kling-Gupta Test: {KGE_test:.3f}")
        print("\n~~~ OTHER STATS ~~~")
        print(f"Train data length: {total_months} months")
        if tuning:
            print(f"\nNumber of trials: {n_trials}")
        print("~~~~~~~~~~~~~~~~~~\n")

        if tuning:
            results.append({
                'Target': target,
                'Best Parameters': str(best_params),
                'RMSE_train': np.round(rmse_train, 3),
                'RMSE_crossval': np.round(rmse_crossval, 3),
                'RMSE_test': np.round(rmse_test, 3),
                'MAE_test': np.round(mae_test, 3),
                'Nash-Sutcliffe Test': np.round(NSE_test, 3),
                'Kling-Gupta Test': np.round(KGE_test, 3),
                'Train data length (months)': total_months,
                'Tune time (s)': np.round(tune_time, 4),
                'Number of trials': n_trials,
                'Elapsed time': elapsed_time
            })

        else:
            results.append({
                'Target': target,
                'Best Parameters': str(best_params),
                'RMSE_train': np.round(rmse_train, 3),
                'RMSE_crossval': np.round(rmse_crossval, 3),
                'RMSE_test': np.round(rmse_test, 3),
                'MAE_test': np.round(mae_test, 3),
                'Nash-Sutcliffe Test': np.round(NSE_test, 3),
                'Kling-Gupta Test': np.round(KGE_test, 3),
                'Train data length (months)': total_months,
                'Inference time (s)': np.round(inference_time, 4),
                'Elapsed time': elapsed_time
            })

    final_time = time.time()
    elapsed_time = final_time - start_time
    print(f"Total elapsed time: {elapsed_time:.4f} seconds\n")
    results.append({'Elapsed time': elapsed_time})
    df_results = pd.DataFrame(results)

    if tuning:
        with open('./models/best_params/GBRT_best_params.pkl', 'wb') as file:
            pickle.dump(params_dict, file)

        df_results.to_csv(f"./models/output/GBRT_tuning_{n_trials}.csv", index=False)
    
    else:
        df_results = pd.DataFrame(results)
        df_results.to_csv("./models/output/GBRT_results_inference.csv", index=False)

    print("Finished")