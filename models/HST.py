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
from processing.custom_metrics import nash_sutcliffe, kling_gupta
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
    
    X.drop(columns=['h'], inplace=True)  # Redundant with h_poly_1
    y.drop(columns=['h'], inplace=True)  # Redundant with h_poly_1

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
    HST_model = LinearRegression()

    tscv = TimeSeriesSplit(n_splits=3, test_size=int(0.1*len(X)))
    cv_scores = cross_val_score(HST_model, X_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rmse_val = np.mean(-cv_scores)
    for idx, score in enumerate(-cv_scores):
        print(f"Split {idx+1}: RMSE = {score}")
    HST_model.fit(X_train, y_train)

    # Predictions
    train_predictions = HST_model.predict(X_train)
    test_predictions = HST_model.predict(X_test)
    all_predictions = HST_model.predict(X_all)

    # Evaluation
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    mae_test = mean_absolute_error(y_test, test_predictions)
    NSE_test = nash_sutcliffe(y_test, test_predictions)
    KGE_test = kling_gupta(y_test, test_predictions)
    feature_names = X_train.columns.tolist()
    coefficients = dict(zip(feature_names, HST_model.coef_))
    return all_predictions, rmse_train, rmse_val, rmse_test, mae_test, NSE_test, KGE_test, coefficients, HST_model


if __name__ == '__main__':
    start_time = time.time()
    # Response_variables = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18']
    Response_variables = ['GV1_extrapolation']
    results = []  # List to store output for each target
    for target in Response_variables:
        start_trial_time = time.time()
        poly_degree = 4
        test_size = 0.3
        n_trials = 2
        X, y, X_train, X_test, y_train, y_test, X_all, split_index, dates, total_months = process_data_for_target(target=target, poly_degree=poly_degree, test_size=test_size)
        start_inference_time = time.time()
        all_predictions, rmse_train, rmse_mean_crossval, rmse_test, mae_test, NSE_test, KGE_test, coefficients, HST_model = hst_train_predict_evaluate(X_train, X_test, y_train, y_test, X_all)
        end_inference_time = time.time()
        plotting_data = {
            'X': X,
            'actual_y': y,
            'X_all': X_all,
            'dates': dates,
            'predictions': all_predictions,
            'split_index': split_index,
            'coefficients': coefficients,
            'RMSE_crossval': rmse_mean_crossval,
            'RMSE': rmse_test,
            'MAE': mae_test,
            'NSE': NSE_test,
            'KGE': KGE_test
        }

        # Pickle: save the plotting data and model to serial files
        with open(f'./visualization/plotting_data/HST/HST_{target}_plotting_data.pkl', 'wb') as f:
            pickle.dump(plotting_data, f)

        with open(f'./visualization/models/HST/HST_model_{target}.pkl', 'wb') as file:
            pickle.dump(HST_model, file)

        print("\n~~~ TARGET ~~~")
        print(target)
        print("\n~~~ TEST METRICS ~~~")
        print(f"RMSE_train: {rmse_train:.3f}")
        print(f"RMSE_crossval: {rmse_mean_crossval:.3f}")
        print(f"RMSE_test: {rmse_test:.3f}")
        print(f"MAE_test: {mae_test:.3f}")
        print(f"Nash-Sutcliffe Test: {NSE_test:.3f}")
        print(f"Kling-Gupta Test: {KGE_test:.3f}")
        print("\n~~~ OTHER STATS ~~~")
        print(f"Train data length: {total_months} months")

        inference_time = end_inference_time - start_inference_time
        print(f"Target inference time: {inference_time:.4f} seconds\n")
        results.append({
            'Target': target,
            'RMSE_train': np.round(rmse_train, 3),
            'RMSE_crossval': np.round(rmse_mean_crossval, 3),
            'RMSE_test': np.round(rmse_test, 3),
            'MAE_test': np.round(mae_test, 3),
            'Nash-Sutcliffe Test': np.round(NSE_test, 3),
            'Kling-Gupta Test': np.round(KGE_test, 3),
            'Train data length (months)': total_months,
            'Target inference time (seconds)': np.round(inference_time, 4)
        })

    final_time = time.time()
    elapsed_time = final_time - start_time
    print(f"Total elapsed time: {elapsed_time:.4f} seconds\n")

    df_results = pd.DataFrame(results)
    # df_results.to_csv("LightGBM_output_results.csv", index=False)

    # Write all metrics to a text file
    with open("./inference_output/HST_inference.txt", "w") as file:
        file.writelines(df_results.to_string(index=False))
        file.write(f"\n\nTotal elapsed time: {elapsed_time:.4f} seconds")

    print("Output saved to HST_inference.txt")