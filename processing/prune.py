# prune.py

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import multiprocessing

N_CORES = multiprocessing.cpu_count()
# For dual process parallelization
HALF_CORES = max(1, N_CORES // 2)
N_JOBS = max(1, N_CORES - 1)


def lgb_feature_prune_rfecv(best_params, X_train, X_test, y_train, y_test, rmse_train, rmse_test, mae_test):
    """
    Performs Recursive Feature Elimination with Cross-Validation (RFECV) to select optimal features
    and compares pruned model performance with the full-feature model.
    
    Parameters:
    -----------
    best_params : dict
        Parameters for LGBMRegressor
    X_train, X_test : DataFrame
        Training and test feature sets
    y_train, y_test : Series
        Training and test target values
    rmse_train, rmse_test, mae_test : float
        Metrics from the full-feature model for comparison
    
    Returns:
    --------
    tuple
        (Pruned model, number of optimal features, pruning_results_dict)
    """
    # Original RFECV code here
    lgbm_rfecv = LGBMRegressor(n_jobs=HALF_CORES, verbose=-1, **best_params)
    n_splits = 3
    cv = TimeSeriesSplit(n_splits=n_splits, test_size=int(0.1*len(X_train)))
    
    rfecv = RFECV(
        estimator=lgbm_rfecv,  #type: ignore
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=HALF_CORES
    )
    rfecv.fit(X_train, y_train)
    optimal_features = rfecv.n_features_
    best_features = X_train.columns[rfecv.support_].tolist()
    
    # Create feature rankings
    feature_ranking = pd.DataFrame({
        'Feature': rfecv.feature_names_in_,
        'Ranking': rfecv.ranking_
    }).sort_values(by='Ranking').reset_index(drop=True)
    
    X_train_pruned = X_train[best_features]
    X_test_pruned = X_test[best_features]

    lgbm_reg_prune = LGBMRegressor(n_jobs=N_JOBS, verbose=-1, **best_params)
    lgbm_reg_prune.fit(X_train_pruned, y_train)
    
    train_preds_pruned = lgbm_reg_prune.predict(X_train_pruned)
    test_preds_pruned = lgbm_reg_prune.predict(X_test_pruned)

    rmse_train_pruned = np.sqrt(mean_squared_error(y_train, train_preds_pruned))
    rmse_test_pruned = np.sqrt(mean_squared_error(y_test, test_preds_pruned))
    mae_test_pruned = mean_absolute_error(y_test, test_preds_pruned)
    
    # Comparison metrics
    comparison = {
        'rmse_train': {'full': rmse_train, 'pruned': rmse_train_pruned, 'improvement': (rmse_train - rmse_train_pruned) / rmse_train * 100},
        'rmse_test': {'full': rmse_test, 'pruned': rmse_test_pruned, 'improvement': (rmse_test - rmse_test_pruned) / rmse_test * 100},
        'mae_test': {'full': mae_test, 'pruned': mae_test_pruned, 'improvement': (mae_test - mae_test_pruned) / mae_test * 100}
    }
    
    # Create a results dictionary to return
    pruning_results = {
        'original_features': len(X_train.columns),
        'optimal_features': optimal_features,
        'best_features': best_features,
        'feature_ranking': feature_ranking,
        'rmse_train_full': rmse_train,
        'rmse_train_pruned': rmse_train_pruned,
        'rmse_train_improvement': comparison['rmse_train']['improvement'],
        'rmse_test_full': rmse_test,
        'rmse_test_pruned': rmse_test_pruned, 
        'rmse_test_improvement': comparison['rmse_test']['improvement'],
        'mae_test_full': mae_test,
        'mae_test_pruned': mae_test_pruned,
        'mae_test_improvement': comparison['mae_test']['improvement']
    }

    # Print the results as in the original function
    print(f"Optimal number of features: {optimal_features}")
    print(f"Selected features: {best_features}")
    print(feature_ranking)
    print(f"New RMSE train: {rmse_train_pruned:.4f}, Old: {rmse_train:.4f}, Improvement: {comparison['rmse_train']['improvement']:.2f}%")
    print(f"New RMSE test: {rmse_test_pruned:.4f}, Old: {rmse_test:.4f}, Improvement: {comparison['rmse_test']['improvement']:.2f}%")
    print(f"New MAE test: {mae_test_pruned:.4f}, Old: {mae_test:.4f}, Improvement: {comparison['mae_test']['improvement']:.2f}%\n\n")
    
    return lgbm_reg_prune, optimal_features, pruning_results


def xgb_feature_prune_rfecv(best_params, X_train, X_test, y_train, y_test, rmse_train, rmse_test, mae_test):
    """
    Performs Recursive Feature Elimination with Cross-Validation (RFECV) to select optimal features
    and compares pruned model performance with the full-feature model.
    
    Parameters:
    -----------
    best_params : dict
        Parameters for XGBRegressor
    X_train, X_test : DataFrame
        Training and test feature sets
    y_train, y_test : Series
        Training and test target values
    rmse_train, rmse_test, mae_test : float, optional
        Metrics from the full-feature model for comparison
    
    Returns:
    --------
    tuple
        (Pruned model, number of optimal features, performance comparison dict)
    """
    
    # RFECV and its estimator both run parallel processes, so split available cores between them
    xgb_rfecv = XGBRegressor(n_jobs=HALF_CORES, **best_params)
    n_splits = 3
    cv = TimeSeriesSplit(n_splits=n_splits, test_size=int(0.1*len(X_train)))
    
    # Perform RFECV - use step parameter for efficiency with many features
    rfecv = RFECV(
        estimator=xgb_rfecv, #type: ignore
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=HALF_CORES
    )
    
    rfecv.fit(X_train, y_train)
    
    optimal_features = rfecv.n_features_
    best_features = X_train.columns[rfecv.support_].tolist()
    
    # Create and display feature rankings
    feature_ranking = pd.DataFrame({
        'Feature': rfecv.feature_names_in_,
        'Ranking': rfecv.ranking_
    }).sort_values(by='Ranking').reset_index(drop=True)
    
    X_train_pruned = X_train[best_features]
    X_test_pruned = X_test[best_features]

    xgb_reg_prune = XGBRegressor(n_jobs=N_JOBS, verbosity=0, **best_params)
    xgb_reg_prune.fit(X_train_pruned, y_train)
    
    train_preds_pruned = xgb_reg_prune.predict(X_train_pruned)
    test_preds_pruned = xgb_reg_prune.predict(X_test_pruned)

    rmse_train_pruned = np.sqrt(mean_squared_error(y_train, train_preds_pruned))
    rmse_test_pruned = np.sqrt(mean_squared_error(y_test, test_preds_pruned))
    mae_test_pruned = mean_absolute_error(y_test, test_preds_pruned)
    
    # Comparison metrics
    comparison = {
        'rmse_train': {'full': rmse_train, 'pruned': rmse_train_pruned, 'improvement': (rmse_train - rmse_train_pruned) / rmse_train * 100},
        'rmse_test': {'full': rmse_test, 'pruned': rmse_test_pruned, 'improvement': (rmse_test - rmse_test_pruned) / rmse_test * 100},
        'mae_test': {'full': mae_test, 'pruned': mae_test_pruned, 'improvement': (mae_test - mae_test_pruned) / mae_test * 100}
    }
    
    # Create a results dictionary to return
    pruning_results = {
        'original_features': len(X_train.columns),
        'optimal_features': optimal_features,
        'best_features': best_features,
        'feature_ranking': feature_ranking,
        'rmse_train_full': rmse_train,
        'rmse_train_pruned': rmse_train_pruned,
        'rmse_train_improvement': comparison['rmse_train']['improvement'],
        'rmse_test_full': rmse_test,
        'rmse_test_pruned': rmse_test_pruned, 
        'rmse_test_improvement': comparison['rmse_test']['improvement'],
        'mae_test_full': mae_test,
        'mae_test_pruned': mae_test_pruned,
        'mae_test_improvement': comparison['mae_test']['improvement']
    }

    # Print the results as in the original function
    print(f"Optimal number of features: {optimal_features}")
    print(f"Selected features: {best_features}")
    print(feature_ranking)
    print(f"New RMSE train: {rmse_train_pruned:.4f}, Old: {rmse_train:.4f}, Improvement: {comparison['rmse_train']['improvement']:.2f}%")
    print(f"New RMSE test: {rmse_test_pruned:.4f}, Old: {rmse_test:.4f}, Improvement: {comparison['rmse_test']['improvement']:.2f}%")
    print(f"New MAE test: {mae_test_pruned:.4f}, Old: {mae_test:.4f}, Improvement: {comparison['mae_test']['improvement']:.2f}%\n\n")
    
    return xgb_reg_prune, optimal_features, pruning_results