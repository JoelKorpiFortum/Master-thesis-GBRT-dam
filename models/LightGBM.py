# LightGBM.py

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
import time
import pickle
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from utils.data_preparation import preprocess_data, split_data
from processing.custom_metrics import willmotts_d, nash_sutcliffe
import multiprocessing

start_time = time.time()

# Response_variables = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18']

features = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18', 'h', \
            'h_MA_007', 'h_MA_014', 'h_RC_007', 'h_RC_030', 'P', 'P_RS_030', \
            'P_RS_060', 'P_RS_090', 'P_RS_180', 'T', 'T_MA_001', 'T_MA_007', \
            't', 'ln_t', 'Cos_s', 'Sin_s', 'Cos_2s', 'Sin_2s', 'month', 'year']

target = 'MB4'
features.remove(target)
poly_degree = 4
start_date = "08-01-2020"
end_date = "03-01-2025"
test_size = 0.3
# split_index = 28107
separation_date = "10-16-2023" # Date corresponds to split_index

X, y, dates = preprocess_data(features, target, start_date, end_date, poly_degree=poly_degree, test_size=test_size)
X_train, X_test, y_train, y_test, split_index = split_data(X, y, test_size=test_size)
X_all = pd.concat([X_train, X_test])

tscv = TimeSeriesSplit(n_splits=3, test_size=int(0.1*len(X)))

param_grid = {
    'n_estimators': [500],     # Number of boosting rounds (equivalent to max_iter in your original code)
    'learning_rate': [0.01],        # Step size at each iteration (your 'learning_rate')
    'max_depth': [3],               # Maximum depth of the tree
    'min_child_weight': [1],        # Minimum sum of instance weight (i.e., the minimum number of samples in a leaf node)
    'subsample': [0.6],             # Fraction of samples to use for training each tree
    'colsample_bytree': [0.8],      # Fraction of features to use for each tree
    'reg_alpha': [3, 5],               # L1 regularization term
    'reg_lambda': [5],         # L2 regularization term (similar to your 'l2_regularization')
    'random_state': [42],           # random state for reproducibility
    'boosting_type': ['gbdt'],      # gbdt or dart, dart takes longer time and works better for more boosting rounds
    'linear_tree' : [True, False]
}

# Initialize the LGBMRegressor
lgb_model = lgb.LGBMRegressor(n_jobs=-multiprocessing.cpu_count() // 2, verbose=-1) # linear_tree=True

# Set up the GridSearchCV with time series split and negative RMSE scoring
grid_search = GridSearchCV(
    estimator=lgb_model, #type: ignore
    param_grid=param_grid, 
    cv=tscv, 
    scoring='neg_root_mean_squared_error', 
    n_jobs=-multiprocessing.cpu_count() // 2,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
model = grid_search.best_estimator_ 

cv_results = grid_search.cv_results_
best_index = grid_search.best_index_
rmse_cv = -cv_results['mean_test_score'][best_index]

# Predictions
train_predictions = model.predict(X_train) #type: ignore
test_predictions = model.predict(X_test) #type: ignore
all_predictions = model.predict(X_all) #type: ignore

# Evaluation
rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
mae_test = mean_absolute_error(y_test, test_predictions)
d_test = willmotts_d(y_test, test_predictions)
NSE_test = nash_sutcliffe(y_test, test_predictions)

# # Compute train/test scores
# train_score = -model.score_
# test_score = np.zeros((model.n_iter_,), dtype=np.float64)
# for i, y_pred in enumerate(model.staged_predict(X_test)):
#     test_score[i] = mean_squared_error(y_test, y_pred)
iterations = np.arange(model.n_iter_) + 1 #type: ignore

# Compute permutation importance
actual_feature_names = X.columns
perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
perm_idx = perm.importances_mean.argsort() #type: ignore
perm_importances = perm.importances[perm_idx].T #type: ignore
perm_labels = actual_feature_names[perm_idx]


fea_imp_ = pd.DataFrame({'cols':X_train.columns, 'fea_imp':model.feature_importances_}) #type: ignore
fea_labels = fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False)

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
#     'train_score': train_score,
#     'test_score': test_score,
    'iterations': iterations,
    'perm_importances': perm_importances,
    'perm_labels':perm_labels,
    'native_importance': fea_imp_,
    'native_labels': fea_labels
}

model_type = 'LightGBM'

# Pickle: save the plotting data and model to serial files
with open(f'./visualization/plotting_data/{model_type}/{model_type}_{target}_plotting_data.pkl', 'wb') as f:
    pickle.dump(plotting_data, f)

with open(f'./visualization/models/{model_type}/{model_type}_model_{target}.pkl', 'wb') as file:
    pickle.dump(model, file)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"RMSE_CV: {rmse_cv}")
print(f"RMSE_train: {rmse_train}")
print(f"RMSE_test: {rmse_test}")
print(f"MAE_test: {mae_test}")
print(f"Willmott's d Test: {d_test}")
print(f"Nash-Sutcliffe Test: {NSE_test}")
print(f"N iter: {model.n_iter_}") #type: ignore
print(f"Best parameters: {best_params}")
print(f"Elapsed time: {elapsed_time:.4f} seconds")