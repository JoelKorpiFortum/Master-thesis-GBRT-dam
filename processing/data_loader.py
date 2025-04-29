# data_loader.py

import pandas as pd
from utils.data_preparation import preprocess_data, split_data, mapping
from dateutil.relativedelta import relativedelta

def process_data_for_target(target, poly_degree=0, test_size=0.2):
    """
    Process data for a given target variable.

    Parameters:
    - target: The target variable to predict.
    - poly_degree: The degree of polynomial features to add (default is 4).
    - test_size: Proportion of data to use as test set (default is 0.3).

    Returns:
    - X_train, X_test, y_train, y_test, split_index, dates, total_months
    """

    non_causal = False
    if non_causal:
        features = ['GV1', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18',
                    'h', 'h_MA_060', 'h_MA_180',
                    'TA', 'TA_MA_180', 'TA_lag_060', 'TA_lag_090', 
                    'P', 'P_RS_180',
                    'Q', 'Q_RS_030', 'Q_RS_120',
                    't', 'month']

    else:
        features = ['h', 'h_MA_060', 'h_MA_180',
                    'TA', 'TA_MA_180', 'TA_lag_060', 'TA_lag_090', 
                    'P', 'P_RS_180',
                    'Q', 'Q_RS_030', 'Q_RS_120',
                    't', 'month']
    
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