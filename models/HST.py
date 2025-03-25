# HST.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from dateutil.relativedelta import relativedelta
from datetime import datetime
import time
import pickle
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from utils.data_preparation import preprocess_data, split_data_normalized, mapping
from processing.custom_metrics import willmotts_d, nash_sutcliffe

start_time = time.time()

# Response_variables = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18']

features = ['h', 'h_poly', 'Cos_2s', 'Sin_2s', 'Sin_s', 'Cos_s', 't', 'ln_t']
target = 'GV1'

path = f'./data/LOS_DAMM_{mapping(target)}.csv'
data = pd.read_csv(path, sep=';', parse_dates=['Date-Time'])

dates = data['Date-Time']
start_date = dates.iloc[0].date()  # First date (YYYY-MM-DD)
end_date = dates.iloc[-1].date()  # Last date (YYYY-MM-DD)

poly_degree = 4
test_size = 0.3
split_index = int(len(data) * (1 - test_size))
separation_date = dates.iloc[split_index].date()
difference = relativedelta(separation_date, start_date)
total_months = difference.years * 12 + difference.months

X, y, dates = preprocess_data(features, target, start_date, end_date, test_size=test_size, poly_degree=poly_degree)
X.drop(columns=['h'], inplace=True)
y.drop(columns=['h'], inplace=True)
X_train, X_test, y_train, y_test, scaler, split_index = split_data_normalized(X, y, test_size=test_size)
X_all = pd.concat([X_train, X_test])

# Fit the model
model = LinearRegression()

tscv = TimeSeriesSplit(n_splits=3, test_size=int(0.1*len(X)))
cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
rmse_cv = np.mean(-cv_scores)
for idx, score in enumerate(-cv_scores):
    print(f"Split {idx+1}: RMSE = {score}")
model.fit(X_train, y_train)

# Predictions 
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
all_predictions = model.predict(X_all)

# Evaluation
rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
mae_test = mean_absolute_error(y_test, test_predictions)
d_test = willmotts_d(y_test, test_predictions)
NSE_test = nash_sutcliffe(y_test, test_predictions)

feature_names = X.columns.tolist()
coefficients = dict(zip(feature_names, model.coef_))

plotting_data = {
    'X_all': X_all,
    'dates': dates,
    'actual_y': y,
    'predictions': all_predictions,
    'split_idx': split_index,
    'coefficients': coefficients,
    'RMSE': rmse_test,
    'MAE': mae_test,
    'WILMOTT': d_test,
    'NSE': NSE_test
}

model_type = 'HST'

# Pickle: save the plotting data and model to serial files
with open(f'./visualization/plotting_data/{model_type}/{model_type}_{target}_plotting_data.pkl', 'wb') as f:
    pickle.dump(plotting_data, f)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"\n~~~ MODEL ~~~ \n")
print(f"Model features: {feature_names}")
print(f"Model Coefficients: {model.coef_}")
print(f"\n~~~ TEST METRICS ~~~ \n")
print(f"RMSE_CV: {rmse_cv}")
print(f"RMSE_train: {rmse_train}")
print(f"RMSE_test: {rmse_test}")
print(f"MAE_test: {mae_test}")
print(f"Willmott's d Test: {d_test}")
print(f"Nash-Sutcliffe Test: {NSE_test}")
print(f"\n~~~ OTHER STATS ~~~ \n")
print(f"Train data length: {total_months} months")
print(f"Elapsed time: {elapsed_time:.4f} seconds\n")