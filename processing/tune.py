#tuning.py

from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm  # progress bar
import multiprocessing

N_CORES = multiprocessing.cpu_count()
# For dual process parallelization
HALF_CORES = max(1, N_CORES // 2)
N_JOBS = max(1, N_CORES - 1)


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

def objective_lgb(trial, X_train, y_train):
    """
    The objective function to tune hyperparameters. It evaluate the score on a
    validation set. This function is used by Optuna, a Bayesian tuning framework.
    """

    params = {
        'random_state': 42,
        'importance_type ': 'gain',
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 22, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 12, log=True),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-1, 100, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-1, 100, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-1, 10, log=True),
        'subsample': trial.suggest_float("subsample", 0.1, 1, log=True),
        # 'linear_tree': trial.suggest_categorical('linear_tree', [True, False]),
        # 'linear_lambda': trial.suggest_float('reg_lambda', 1e-8, 100, log=True)
    }

    model = LGBMRegressor(objective='regression', n_jobs=N_JOBS, verbose=-1, **params)
    cv_rmse = tscv_model(model, X_train, y_train)
    del model
    return np.mean(cv_rmse)

def lgb_tune(target, X_train, y_train, n_trials):
    # Uncomment to suppress optuna log messages
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42)
    study_model = optuna.create_study(direction='minimize', sampler=sampler, study_name=f'hyperparameter_tuning_{target}')
    study_model.optimize(lambda trial: objective_lgb(trial, X_train, y_train), n_trials=n_trials)  # type: ignore

    trial = study_model.best_trial
    best_params = trial.params
    print('Best params from optuna: \n', best_params)
    del study_model
    return best_params


def objective_xgb(trial, X_train, y_train):
    """
    The objective function to tune hyperparameters. It evaluate the score on a
    validation set. This function is used by Optuna, a Bayesian tuning framework.
    """

    params = {
            'random_state': 42,
            'tree_method': 'hist',
            'importance_type': 'gain',
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'learning_rate': trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 12),
            'max_leaves': trial.suggest_int('max_leaves', 2, 22),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1, log=True),
            'subsample': trial.suggest_float('subsample', 0.1, 1, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha',  1e-1, 50, log=True),  
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-1, 50, log=True), 
            'gamma': trial.suggest_float('gamma', 1e-1, 10, log=True)
    }

    model = XGBRegressor(objective='reg:squarederror', n_jobs=N_JOBS, verbosity=0, **params)
    cv_rmse = tscv_model(model, X_train, y_train)
    del model
    return np.mean(cv_rmse)


def xgb_tune(target, X_train, y_train, n_trials):
    # Uncomment to suppress optuna log messages
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42)
    study_model = optuna.create_study(direction='minimize', sampler=sampler, study_name=f'hyperparameter_tuning_{target}')
    study_model.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=n_trials)  # type: ignore

    trial = study_model.best_trial
    best_params = trial.params
    print('Best params from optuna: \n', best_params)
    del study_model
    return best_params

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
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 22),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-1, 20, log=True),
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