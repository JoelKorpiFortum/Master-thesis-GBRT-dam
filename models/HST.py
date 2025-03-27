# HST.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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

    features = ['h', 'h_poly', 'Cos_2s', 'Sin_2s', 'Sin_s', 'Cos_s', 't', 'ln_t']
    # features.remove(target)
    path = f'./data/LOS_DAMM_{mapping(target)}.csv'
    data = pd.read_csv(path, sep=';', parse_dates=['Date-Time'])

    # Extract start and end dates
    dates = data['Date-Time']
    start_date = dates.iloc[0].date()  # First date (YYYY-MM-DD)
    end_date = dates.iloc[-1].date()  # Last date (YYYY-MM-DD)

    # Preprocess the data (You would need to define preprocess_data elsewhere in your code)
    X, y, dates = preprocess_data(features, target, start_date, end_date, poly_degree=poly_degree, test_size=test_size)
    
    X.drop(columns=['h'], inplace=True)
    y.drop(columns=['h'], inplace=True)

    # Split the data into training and testing sets (Assumes split_data is defined elsewhere)
    X_train, X_test, y_train, y_test, split_index, _ = split_data_normalized(X, y, test_size=test_size)

    # Concatenate training and test data for full data (if needed)
    X_all = pd.concat([X_train, X_test])

    # Calculate the separation date between training and test data
    separation_date_train = dates.iloc[split_index].date()
    difference = relativedelta(separation_date_train, start_date)
    total_months = difference.years * 12 + difference.months

    # Return the processed results
    return X, y, X_train, X_test, y_train, y_test, X_all, split_index, dates, total_months


def hst_train_predict_evaluate(X_train, X_test, y_train, y_test, X_all):
    # Fit model
    model = LinearRegression()

    tscv = TimeSeriesSplit(n_splits=3, test_size=int(0.1*len(X)))
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rmse_val = np.mean(-cv_scores)
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
    mae_test = mean_absolute_error(y_test, test_predictions)
    d_test = willmotts_d(y_test, test_predictions)
    NSE_test = nash_sutcliffe(y_test, test_predictions)
    feature_names = X_train.columns.tolist()
    coefficients = dict(zip(feature_names, model.coef_))
    return all_predictions, rmse_train, rmse_val, rmse_test, mae_test, d_test, NSE_test, coefficients


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
        all_predictions, rmse_train, rmse_crossval, rmse_test, mae_test, d_test, NSE_test, coefficients = hst_train_predict_evaluate(X_train, X_test, y_train, y_test, X_all)

        plotting_data = {
            'X': X,
            'actual_y': y,
            'X_all': X_all,
            'dates': dates,
            'predictions': all_predictions,
            'split_index': split_index,
            'coefficients': coefficients,
            'RMSE_crossval': rmse_crossval,
            'RMSE': rmse_test,
            'MAE': mae_test,
            'WILMOTT': d_test,
            'NSE': NSE_test
        }

        # Pickle: save the plotting data and model to serial files
        with open(f'./visualization/plotting_data/HST/HST_{target}_plotting_data.pkl', 'wb') as f:
            pickle.dump(plotting_data, f)

        print("\n~~~ TARGET ~~~")
        print(target)
        print("\n~~~ TEST METRICS ~~~")
        print(f"RMSE_train: {rmse_train:.3f}")
        print(f"RMSE_crossval: {rmse_crossval:.3f}")
        print(f"RMSE_test: {rmse_test:.3f}")
        print(f"MAE_test: {mae_test:.3f}")
        print(f"Willmott's d Test: {d_test:.3f}")
        print(f"Nash-Sutcliffe Test: {NSE_test:.3f}")
        print("\n~~~ OTHER STATS ~~~")
        print(f"Train data length: {total_months:.3f} months")

        # Append each piece of information to the output list
        output_lines.append(f"RMSE_train: {rmse_train:.3f}\n")
        output_lines.append(f"RMSE_crossval: {rmse_crossval:.3f}\n")
        output_lines.append(f"RMSE_test: {rmse_test:.3f}\n")
        output_lines.append(f"MAE_test: {mae_test:.3f}\n")
        output_lines.append(f"Willmott's d Test: {d_test:.3f}\n")
        output_lines.append(f"Nash-Sutcliffe Test: {NSE_test:.3f}\n")
        output_lines.append(f"Train data length: {total_months:} months\n")

        end_trial_time = time.time()
        trial_time = end_trial_time - start_trial_time
        print(f"Trial time: {trial_time:.4f} seconds\n")
        output_lines.append(f"Trial time: {trial_time:.4f} seconds\n")
        output_lines.append("\n")  # Add an empty line for separation

    final_time = time.time()
    elapsed_time = final_time - start_time
    print(f"Total elapsed time: {elapsed_time:.4f} seconds\n")
    output_lines.append(f"Total elapsed time: {elapsed_time:.4f} seconds\n\n")

    # Write all metrics to a text file
    with open("HST_output.txt", "w") as file:
        file.writelines(output_lines)

    print("Output saved to HST_output.txt")