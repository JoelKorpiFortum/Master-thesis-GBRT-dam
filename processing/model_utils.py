# processing/model_utils.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import pickle
import logging
import multiprocessing
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from processing.custom_metrics import nash_sutcliffe, kling_gupta
from processing.tune import tscv_model, lgb_tune, xgb_tune, gbrt_tune
from processing.prune import lgb_feature_prune_rfecv, xgb_feature_prune_rfecv

# Configure multiprocessing
N_CORES = multiprocessing.cpu_count()
N_JOBS = max(1, N_CORES - 1)

# Model class mapping
MODEL_CLASSES = {
    'LightGBM': LGBMRegressor,
    'XGBoost': XGBRegressor,
    'GBRT': HistGradientBoostingRegressor
}

# Tuning function mapping
TUNE_FUNCTIONS = {
    'LightGBM': lgb_tune,
    'XGBoost': xgb_tune,
    'GBRT': gbrt_tune
}

# Pruning function mapping
PRUNE_FUNCTIONS = {
    'LightGBM': lgb_feature_prune_rfecv,
    'XGBoost': xgb_feature_prune_rfecv,
    'GBRT': None  # GBRT doesn't have pruning, lack of support for HistGradientBooster
}


def get_best_params(model_name, target):
    """Load best parameters from pickled file for given target and model."""
    with open(f'./models/best_params/{model_name}_best_params.pkl', 'rb') as f:
        best_params_dict = pickle.load(f)
    return best_params_dict.get(target, None)


def get_saved_params(model_name, target):
    """Get saved parameters from the HYPERPARAMETERS dictionary."""
    # Dynamically import the appropriate module
    module_name = f"models.best_params.{model_name}_saved_params"
    module = __import__(module_name, fromlist=['HYPERPARAMETERS'])
    HYPERPARAMETERS = getattr(module, 'HYPERPARAMETERS')
    
    return HYPERPARAMETERS.get(target, None)


def predict_evaluate(model_name, best_params, X_train, X_test, y_train, y_test, X_all):
    """Generic function to fit model, make predictions, and evaluate performance."""
    # Get the model class
    model_class = MODEL_CLASSES[model_name]
    
    # Create and fit model with appropriate parameters
    if model_name == 'LightGBM':
        opt_model = model_class(**best_params, verbose=-1, n_jobs=N_JOBS)
    elif model_name == 'XGBoost':
        opt_model = model_class(**best_params, n_jobs=N_JOBS)
    elif model_name == 'GBRT':
        opt_model = model_class(verbose=0, random_state=42, **best_params)
    
    opt_model.fit(X_train, y_train)

    # Predictions
    train_predictions = opt_model.predict(X_train)
    test_predictions = opt_model.predict(X_test)
    all_predictions = opt_model.predict(X_all)

    # Evaluation
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
    rmse_crossval_folds = tscv_model(opt_model, X_train, y_train) # Comment out for inference tests
    rmse_crossval = np.mean(rmse_crossval_folds)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    mae_test = mean_absolute_error(y_test, test_predictions)
    NSE_test = nash_sutcliffe(y_test, test_predictions)
    KGE_test = kling_gupta(y_test, test_predictions)

    return opt_model, all_predictions, rmse_train, rmse_crossval, rmse_test, mae_test, NSE_test, KGE_test


def save_results(model_name, results, tuning=False, prune=False, n_trials=None):
    """Save model results to CSV file."""
    df_results = pd.DataFrame(results)
    
    if tuning and n_trials:
        df_results.to_csv(f"./models/output/{model_name}_tuning_{n_trials}.csv", index=False)
    elif not prune:
        df_results.to_csv(f"./models/output/{model_name}_results_inference.csv", index=False)


def save_tuning_params(model_name, params_dict):
    """Save tuned hyperparameters to pickle file."""
    with open(f'./models/best_params/{model_name}_best_params.pkl', 'wb') as file:
        pickle.dump(params_dict, file)


def save_model_and_data(model_name, target, best_params, plotting_data):
    """Save model and plotting data for visualization."""
    # Save plotting data
    with open(f'./visualization/plotting_data/{model_name}/{model_name}_{target}_plotting_data.pkl', 'wb') as f:
        pickle.dump(plotting_data, f)

    # Save model
    model_class = MODEL_CLASSES[model_name]
    if model_name == 'LightGBM':
        opt_model = model_class(**best_params, verbose=-1)
    elif model_name == 'XGBoost':
        opt_model = model_class(**best_params)
    elif model_name == 'GBRT':
        opt_model = model_class(**best_params)
    
    with open(f'./visualization/models/{model_name}/{model_name}_model_{target}.pkl', 'wb') as file:
        pickle.dump(opt_model, file)


def save_pruning_results(model_name, pruning_results, targets):
    """Save pruning results to CSV file."""
    pruning_df = pd.DataFrame([
        {
            'target': res['target'],
            'original_features': res['original_features'],
            'optimal_features': res['optimal_features'], 
            'rmse_train_full': res['rmse_train_full'],
            'rmse_train_pruned': res['rmse_train_pruned'],
            'rmse_train_improvement_%': res['rmse_train_improvement'],
            'rmse_test_full': res['rmse_test_full'],
            'rmse_test_pruned': res['rmse_test_pruned'],
            'rmse_test_improvement_%': res['rmse_test_improvement'],
            'mae_test_full': res['mae_test_full'], 
            'mae_test_pruned': res['mae_test_pruned'],
            'mae_test_improvement_%': res['mae_test_improvement'],
            'selected_features': ';'.join(res['best_features'])
        } for res in pruning_results
    ])

    # Save to CSV
    pruning_df.to_csv(f'./models/output/{model_name}_rfecv_pruning_results.csv', index=False)
    logging.info(f"Pruning results for all targets saved")

    # Save detailed feature rankings for each target
    for res in pruning_results:
        target = res['target']
        rankings = res['feature_ranking']
        rankings['target'] = target

        # Append to a single file or create separate files per target
        if target == targets[0]:  # First target, create new file
            rankings.to_csv(f'./models/output/{model_name}_feature_rankings_all_targets.csv', index=False)
        else:  # Append to existing file
            rankings.to_csv(f'./models/output/{model_name}_feature_rankings_all_targets.csv', 
                          mode='a', header=False, index=False)


def process_target(model_name, target, poly_degree, test_size, tuning, n_trials, prune):
    """Generic process for training, evaluating, and optionally tuning/pruning a target."""
    logger = logging.getLogger(__name__)
    
    # Load and process data
    from processing.data_loader import process_data_for_target
    X, y, X_train, X_test, y_train, y_test, X_all, split_index, dates, total_months = process_data_for_target(
        target=target, poly_degree=poly_degree, test_size=test_size
    )
    
    # Get parameters (either tune or load)
    if tuning:
        start_tune_time = time.time()
        tune_func = TUNE_FUNCTIONS[model_name]
        best_params = tune_func(target, X_train, y_train, n_trials)
        tune_time = time.time() - start_tune_time
        logger.info(f"Target tuning time: {tune_time:.4f} seconds")
    else:
        best_params = get_saved_params(model_name, target)
    
    # Predict and evaluate
    start_inference_time = time.time()
    opt_model, all_predictions, rmse_train, rmse_crossval, rmse_test, mae_test, NSE_test, KGE_test = predict_evaluate(
        model_name, best_params, X_train, X_test, y_train, y_test, X_all
    )
    inference_time = time.time() - start_inference_time
    logger.info(f"Target inference time: {inference_time:.4f} seconds")
    
    # Create plotting data dictionary
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
    
    # Handle pruning if requested
    target_pruning_results = None
    if prune and PRUNE_FUNCTIONS[model_name]:
        logger.info(f"Pruning target: {target}")
        prune_func = PRUNE_FUNCTIONS[model_name]
        model_prune, optimal_features, target_pruning_results = prune_func(
            best_params, X_train, X_test, y_train, y_test, rmse_train, rmse_test, mae_test
        )
        # Add target name to the results
        target_pruning_results['target'] = target
        
        # Save pruned model
        with open(f'./visualization/models/{model_name}/{model_name}_prune_model_{target}.pkl', 'wb') as file:
            pickle.dump({'model': model_prune, 'features': optimal_features}, file)
    else:
        # Log results
        logger.info(f"\n\n~~~ TARGET: {target} ~~~")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"RMSE_train: {rmse_train:.3f}")
        logger.info(f"RMSE_crossval: {rmse_crossval:.3f}")
        logger.info(f"RMSE_test: {rmse_test:.3f}")
        logger.info(f"MAE_test: {mae_test:.3f}")
        logger.info(f"Nash-Sutcliffe Test: {NSE_test:.3f}")
        logger.info(f"Kling-Gupta Test: {KGE_test:.3f}")
        if tuning:
            logger.info(f"Number of trials: {n_trials}")
    
    # Create result dictionary
    result = {
        'Target': target,
        'Best Parameters': str(best_params),
        'RMSE_train': np.round(rmse_train, 3),
        'RMSE_crossval': np.round(rmse_crossval, 3),
        'RMSE_test': np.round(rmse_test, 3),
        'MAE_test': np.round(mae_test, 3),
        'Nash-Sutcliffe Test': np.round(NSE_test, 3),
        'Kling-Gupta Test': np.round(KGE_test, 3),
        'Train data length (months)': total_months,
    }
    
    # Add specific fields based on mode
    if tuning:
        result.update({
            'Tune time (s)': np.round(tune_time, 4),
            'Number of trials': n_trials,
        })
    else:
        result.update({
            'Inference time (s)': np.round(inference_time, 4),
        })
    
    return best_params, plotting_data, result, target_pruning_results