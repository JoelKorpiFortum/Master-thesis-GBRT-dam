# SeasonalNaive.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dateutil.relativedelta import relativedelta
from datetime import datetime
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from processing.custom_metrics import willmotts_d, nash_sutcliffe
from utils.data_preparation import mapping
import pickle

# Response_variables = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18']

target = 'GV1'
path = f'./data/LOS_DAMM_{mapping(target)}.csv'
data = pd.read_csv(path, sep=';', parse_dates=['Date-Time'])
print(data)

dates = data['Date-Time']
start_date = dates.iloc[0].date()  # First date (YYYY-MM-DD)
end_date = dates.iloc[-1].date()  # Last date (YYYY-MM-DD)

test_size = 0.3
split_index = int(len(data) * (1 - test_size))
separation_date = dates.iloc[split_index].date()
difference = relativedelta(separation_date, start_date)
total_months = difference.years * 12 + difference.months

lag_num = 24*365 # Seasonality = amount of hours (data points) in a year
test_size = 0.3 # Same split as other methods

# Split data based on split_index
X_train = data[:split_index]
X_test = data.iloc[split_index:]
X_all = pd.concat([X_train, X_test])

y_train = X_train[mapping(target)]
y_test = X_test[mapping(target)]
y_all = pd.concat([y_train, y_test])

# Seasonal Naive Forecasting Model (using .shift() to shift by one year)
seasonal_naive_model = y_all.copy()
seasonal_naive_predictions = seasonal_naive_model.shift(lag_num)
seasonal_naive_predictions = seasonal_naive_predictions.iloc[split_index:]
y_test_clean = y_test.loc[seasonal_naive_predictions.index]

# Non-test values of model
seasonal_naive_model_nontest = seasonal_naive_model.shift(lag_num).iloc[:split_index]

# Evaluation
rmse_test = np.sqrt(mean_squared_error(y_test_clean, seasonal_naive_predictions))
rmse_test = np.sqrt(mean_squared_error(y_test_clean, seasonal_naive_predictions))
mae_test = mean_absolute_error(y_test_clean, seasonal_naive_predictions)
d_test = willmotts_d(y_test_clean, seasonal_naive_predictions)
NSE_test = nash_sutcliffe(y_test_clean, seasonal_naive_predictions)

plotting_data = {
    'X_all': X_all,
    'dates': dates,
    'actual_y': y_all,
    'predictions': seasonal_naive_predictions,
    'non_predicitons': seasonal_naive_model_nontest,
    'split_idx': split_index,
    'RMSE': rmse_test,
    'MAE': mae_test,
    'WILMOTT': d_test,
    'NSE': NSE_test
}

model_type = 'Naive'

with open(f'./visualization/plotting_data/{model_type}/{model_type}_{target}_plotting_data.pkl', 'wb') as f:
    pickle.dump(plotting_data, f)

print(f"\n~~~ TEST METRICS ~~~ \n")
print(f"RMSE_test: {rmse_test}")
print(f"MAE_test: {mae_test}")
print(f"Willmott's d Test: {d_test}")
print(f"Nash-Sutcliffe Test: {NSE_test}")
print(f"\n~~~ OTHER STATS ~~~ \n")
print(f"Train data length: {total_months} months")