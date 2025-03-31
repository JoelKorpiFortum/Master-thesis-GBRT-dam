# data_preparation.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FILE_PATHS = {
    'ÖVY': './data/LOS_DAMM_ÖVY.csv',
    'MB1_AM_L1': './data/LOS_DAMM_MB1_AM_L1.csv',
    'MB1_AM_L1_extrapolation': './data/LOS_DAMM_MB1_AM_L1_extrapolation.csv',
    'MB1_AM_L3': './data/LOS_DAMM_MB1_AM_L3.csv',
    'MB1_HM_L51': './data/LOS_DAMM_MB1_HM_L51.csv',
    'MB4_A_D': './data/LOS_DAMM_MB4_A_D.csv',
    'MB8_A_D': './data/LOS_DAMM_MB8_A_D.csv',
    'MB8_A_D_LS': './data/LOS_DAMM_MB8_A_D_LS.csv',
    'MB10_A_D': './data/LOS_DAMM_MB10_A_D.csv',
    'MB10_A_D_LS': './data/LOS_DAMM_MB10_A_D_LS.csv',
    'MB18_A_D': './data/LOS_DAMM_MB18_A_D.csv',
    'UTETEMP': './data/LOS_DAMM_UTETEMP.csv',
    'PRECIP': './data/LOS_DAMM_PRECIP.csv',
}

DATASET_MAPPING = {
    'h': 'ÖVY',
    'h_MA_007': 'derived_temporal', # MA = MOVING AVERAGE
    'h_MA_014': 'derived_temporal', # MA
    'h_MA_030': 'derived_temporal', # MA
    'h_MA_060': 'derived_temporal', # MA
    'h_RC_007': 'derived_temporal', # RC = RATE OF CHANGE
    'h_RC_030': 'derived_temporal', # RC
    'GV1': 'MB1_AM_L1',
    'GV1_extrapolation': 'MB1_AM_L1_extrapolation',
    'GV3': 'MB1_AM_L3',
    'GV51': 'MB1_HM_L51',
    'MB4': 'MB4_A_D',
    # 'MB8': 'MB8_A_D',
    'MB8': 'MB8_A_D_LS',
    # 'MB10': 'MB10_A_D',
    'MB10': 'MB10_A_D_LS',
    'MB18': 'MB18_A_D',
    'T': 'UTETEMP',
    'T_MA_001': 'derived_temporal', # MA
    'T_MA_007': 'derived_temporal', # MA
    'T_MA_014': 'derived_temporal', # MA
    'T_MA_030': 'derived_temporal', # MA
    'T_MA_060': 'derived_temporal',
    'P': 'PRECIP',
    'P_RS_030': 'derived_temporal', # RS = RUNNING SUM
    'P_RS_060': 'derived_temporal', # RS 
    'P_RS_090': 'derived_temporal', # RS
    'P_RS_180': 'derived_temporal', # RS
    'h_poly': 'derived',
    'Sin_s': 'derived', 
    'Cos_s': 'derived',  
    'Sin_2s': 'derived',     
    'Cos_2s': 'derived',     
    't': 'derived',       
    'ln_t': 'derived', 
    'month': 'derived',
    'year': 'derived',
    'feature_name': 'dataset_key',
}

DERIVED_FEATURE_BASES = {
    'h_MA_007': 'h',
    'h_MA_014': 'h',
    'h_MA_030': 'h',
    'h_MA_060': 'h',
    'h_MA_180': 'h',
    'h_RC_007': 'h',
    'h_RC_030': 'h',
    'T_MA_001': 'T',
    'T_MA_007': 'T',
    'T_MA_014': 'T',
    'T_MA_030': 'T',
    'T_MA_060': 'T',
    'P_RS_030': 'P',
    'P_RS_060': 'P',
    'P_RS_090': 'P',
    'P_RS_180': 'P',
    'h_poly': 'h'
}


def load_and_split_dataset(path, common_start_date=None, common_end_date=None, validation_size=0.2, test_size=0.2):
    """ Helper function ot load, align and interpolate measurements to create one uniform dataset."""

    df = pd.read_csv(path, sep=';', parse_dates=['Date-Time']) # Forgot to add sep
    df.set_index('Date-Time', inplace=True)
    df = df.groupby(df.index).mean()  # Aggregate duplicates (daylight saving time change)

    # Match starting dates
    if common_start_date:
        start_date = pd.to_datetime(common_start_date)
    else:
        start_date = df.index.min()
    
    if common_end_date:
        end_date = pd.to_datetime(common_end_date)
    else:
        end_date = df.index.max()
        
    # Reindex to hourly frequency
    all_dates = pd.date_range(start=start_date, end=end_date, freq='h')
    df_reindexed = df.reindex(all_dates)

    # Split into train, validation and test
    split_idx_train = int(len(df_reindexed) * (1 - test_size - validation_size))
    split_idx_validation = int(len(df_reindexed) * (1 - test_size))
    
    train = df_reindexed.iloc[:split_idx_train]
    validation = df_reindexed.iloc[split_idx_train:split_idx_validation]
    test = df_reindexed.iloc[split_idx_validation:]

    # Interpolate separately on train, validation and test sets
    train_interpolated = train.interpolate(method='time')
    validation_interpolated = validation.interpolate(method='time')
    test_interpolated = test.interpolate(method='time')

    # Combine interpolated sets
    df_interpolated = pd.concat([train_interpolated, validation_interpolated, test_interpolated])
    df_interpolated.reset_index(inplace=True)
    df_interpolated.rename(columns={'index': 'Date-Time'}, inplace=True)
    
    return df_interpolated, split_idx_train, split_idx_validation


def load_and_merge_data(features, target, start_date, end_date, validation_size=0.2, test_size=0.2):
    """ Combines all measurements into one dataset."""

    datasets_required = set()
    for variable in features + [target]:
        if variable in DATASET_MAPPING:
            dataset = DATASET_MAPPING[variable]
            if 'derived' not in dataset:
                datasets_required.add(dataset)
            else:
                base_variable = DERIVED_FEATURE_BASES.get(variable)
                if base_variable:
                    base_dataset = DATASET_MAPPING.get(base_variable)
                    if 'derived' not in dataset and base_dataset:
                        datasets_required.add(base_dataset)
        else:
            raise ValueError(f"Variable '{variable}' not found in DATASET_MAPPING.")
    
    loaded_dataframes = {}
    for dataset_key in datasets_required:
        path = FILE_PATHS.get(dataset_key)
        if path is None:
            raise ValueError(f"Dataset '{dataset_key}' not found in FILE_PATHS.")
        df_interpolated, split_idx_train, split_idx_validation = load_and_split_dataset(path, common_start_date=start_date, \
                                                  common_end_date=end_date, test_size=test_size, validation_size=validation_size)
        loaded_dataframes[dataset_key] = df_interpolated

    df_merged = pd.DataFrame()
    for df in loaded_dataframes.values():
        if df_merged.empty:
            df_merged = df
        else:
            df_merged = pd.merge(df_merged, df, on='Date-Time', how='inner')

    return df_merged, split_idx_train, split_idx_validation


def apply_transformation(data, base_var, transformation_type, window_size):
    """ Helper function to apply the transformation on train or test data separately."""
    if transformation_type == 'RS':  # Running sum
        return data[base_var].rolling(window=window_size, center=False, min_periods=1).sum() #Center=false ensures only current data is used
    elif transformation_type == 'MA':  # Moving average
        return data[base_var].rolling(window=window_size, center=False, min_periods=1).mean()
    elif transformation_type == 'RC':  # Rate of change
        rolling_mean = data[base_var].rolling(window=window_size, center=False, min_periods=1).mean() #for stability
        return rolling_mean.pct_change(periods=window_size) * 100


def preprocess_temporal_features(data, feature, split_idx_train, split_idx_validation):
    """ Function for derived temporal features, used for avoiding 
        data leakage by separating calculations for train and test data."""

    data_preprocessed_train = pd.DataFrame()
    data_preprocessed_val = pd.DataFrame()
    data_preprocessed_test = pd.DataFrame()
    
    # Separate the train and test data
    data_train = data.iloc[:split_idx_train]
    data_val = data.iloc[split_idx_train:split_idx_validation]
    data_test = data.iloc[split_idx_validation:]

    data_preprocessed_train = data_train[['Date-Time']].copy()
    data_preprocessed_val = data_val[['Date-Time']].copy()
    data_preprocessed_test = data_test[['Date-Time']].copy()

    # Split string to get each component of the specified feature
    split_string = feature.split('_')
    base_var = split_string[0] # e.g: h
    base_var = DATASET_MAPPING.get(base_var) # e.g: h => ÖVY
    transformation_type = split_string[1]  # e.g: MA
    window_size = int(split_string[2]) * 24  # e.g: 7*24, converts to hours

    # Apply transformation for train and test using helper function
    data_preprocessed_train[feature] = apply_transformation(data_train, base_var, transformation_type, window_size)
    data_preprocessed_val[feature] = apply_transformation(data_val, base_var, transformation_type, window_size)
    data_preprocessed_test[feature] = apply_transformation(data_test, base_var, transformation_type, window_size)

    # Concat train and test dataframes before returning
    data_preprocessed = pd.concat([data_preprocessed_train, data_preprocessed_val, data_preprocessed_test], axis=0).sort_index()
    data_preprocessed = data_preprocessed[[feature]]

    return data_preprocessed

def preprocess_data(features, target, start_date, end_date, poly_degree=None, validation_size=0.2, test_size=0.2):
    """ Fetches data and creates all derived features."""

    df, split_idx_train, split_idx_validation = load_and_merge_data(features, target, start_date, end_date, validation_size=validation_size, test_size=test_size)
    target_name = DATASET_MAPPING[target]
    df_preprocessed = pd.DataFrame()
    df_preprocessed = df[['Date-Time', target_name]].copy()

    if any(feature in features for feature in ['Sin_s', 'Cos_s', 'Sin_2s', 'Cos_2s']):
        day_of_year = df['Date-Time'].dt.dayofyear
        s = 2 * np.pi * day_of_year / 365.0
        if 'Sin_s' in features:
            df_preprocessed['Sin_s'] = np.sin(s)
        if 'Cos_s' in features:
            df_preprocessed['Cos_s'] = np.cos(s)
        if 'Sin_2s' in features:
            df_preprocessed['Sin_2s'] = np.sin(2 * s)
        if 'Cos_2s' in features:
            df_preprocessed['Cos_2s'] = np.cos(2 * s)
    
    if 't' or 'ln_t' in features:
        min_date = df['Date-Time'].min()
        df_preprocessed['t'] = (df['Date-Time'] - min_date).dt.days + 1
        if 'ln_t' in features:
            df_preprocessed['ln_t'] = np.log(df_preprocessed['t'])
    
    if 'month' in features:
        df_preprocessed['month'] = df_preprocessed['Date-Time'].dt.month
    if 'year' in features:
        df_preprocessed['year'] = df_preprocessed['Date-Time'].dt.year

    for feature in features:
        feature_origin = DATASET_MAPPING[feature]
        if feature_origin == 'derived': # Simple derived features
            if '_lag_' in feature:
                base_var, lag_num = feature.split('_lag_')
                lag_num = int(lag_num)
                df_preprocessed[feature] = df[base_var].shift(lag_num)
            elif feature == 'h_poly' and poly_degree is not None:
                poly = PolynomialFeatures(degree=poly_degree)
                h_poly = poly.fit_transform(df[['ÖVY']])
                for i in range(1, poly_degree + 1):
                    df_preprocessed[f'h_poly_{i}'] = h_poly[:, i]

        elif feature_origin == 'derived_temporal': # Temporal derived features
            df_preprocessed[feature] = preprocess_temporal_features(df, feature, split_idx_train, split_idx_validation)

        elif feature_origin in df: # Non-derived features, no processing
            df_preprocessed[feature] = df[feature_origin]

    # df_preprocessed.dropna(inplace=True)
    dates = df_preprocessed['Date-Time']
    y = df_preprocessed[target_name]
    X = df_preprocessed.drop(columns=['Date-Time', target_name])

    return X, y, dates


def split_data(X, y, test_size=0.2, split_index_from_end=None):
    """ Splits data into train and test WITHOUT normalization 
    (not necessary for boosted trees)."""

    if split_index_from_end:
        split_idx = len(y) - split_index_from_end
    else:
        split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx] if y is not None else None
    y_test = y.iloc[split_idx:] if y is not None else None

    return X_train, X_test, y_train, y_test, split_idx


def split_data_normalized(X, y, test_size=0.2, split_index_from_end=None):
    """ Splits data into train and test WITH normalization 
    (necessary for HST)."""

    if split_index_from_end:
        split_idx = len(y) - split_index_from_end
    else:
        split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx] if y is not None else None
    y_test = y.iloc[split_idx:] if y is not None else None

    # Normalize for train and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, split_idx, scaler


def split_data_validation(X, y, validation_size=0.2, test_size=0.2, split_index_from_end=None):
    """ Splits data into train, validation and test 
    WITHOUT normalization."""

    if split_index_from_end:
        split_idx = len(y) - split_index_from_end
    else:
        split_idx_validation = int(len(X) * (1 - test_size - validation_size))
        split_idx_test = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx_validation]
    X_val = X.iloc[split_idx_validation:split_idx_test]
    X_test = X.iloc[split_idx_test:]
    y_train = y.iloc[:split_idx_validation] if y is not None else None
    y_val = y.iloc[split_idx_validation:split_idx_test] if y is not None else None
    y_test = y.iloc[split_idx_test:] if y is not None else None

    return X_train, X_val, X_test, y_train, y_val, y_test, split_idx_validation, split_idx_test


def mapping(feature):
    """ Returns correctly mapped feature name from original signal."""

    return DATASET_MAPPING[feature]


if __name__ == "__main__":
    features = ['h_poly', 'h', 'GV1', 'GV1_extrapolation', 'GV3', 'GV51', 'MB4', 'MB8', 'MB10', 'MB18', \
                 'P', 'T', 't', 'ln_t', 'Cos_s', 'Sin_s', 'Cos_2s', 'Sin_2s', 'month', 'year', \
                 'h_MA_007', 'h_MA_014', 'h_MA_030', 'h_MA_060', 'h_RC_007', 'h_RC_030', 'T_MA_001', 'T_MA_007', \
                 'T_MA_030', 'T_MA_060', 'P_RS_030', 'P_RS_060', 'P_RS_090', 'P_RS_180']
    poly_degree = 4
    target = 'GV3'
    start_date = "08-01-2020" #MM-dd-YYYY
    end_date = "03-01-2025" #MM-dd-YYYY
    validation_size = 0
    test_size = 0.2
    
    X, y, dates = preprocess_data(features, target, start_date, end_date, poly_degree=poly_degree, \
                               test_size=test_size, validation_size=validation_size)
    
    # X_train, X_val, X_test, y_train, y_val, y_test, split_idx_validation, split_idx_test = split_data_validation(X, y, validation_size=validation_size, test_size=test_size)
    X_train, X_test, y_train, y_test, split_idx = split_data(X, y, test_size=test_size)
    print("Data preparation completed.")


    """ Optional: create a file to write the output""" 
    with open("output.txt", "w") as file:
        # Writing the outputs of X, y, and other variables to the file
        file.write("X (Features):\n")
        file.write(str(X) + "\n\n")  # Convert to string so it can be written to the file
        file.write("y (Target):\n")
        file.write(str(y) + "\n\n")
        file.write("X_train (Training Features):\n")
        file.write(str(X_train) + "\n\n")
        # file.write("X_val (Validation Features):\n")
        # file.write(str(X_val) + "\n\n")
        file.write("X_test (Test Features):\n")
        file.write(str(X_test) + "\n\n")
        file.write("y_train (Training Target):\n")
        file.write(str(y_train) + "\n\n")
        # file.write("y_val (Validation Target):\n")
        # file.write(str(y_val) + "\n\n")
        file.write("y_test (Test Target):\n")
        file.write(str(y_test) + "\n\n")
        # file.write("Split Index Calidaiton:\n")
        # file.write(str(split_idx_validation) + "\n\n")
        # file.write("Split Index Test):\n")
        # file.write(str(split_idx_test) + "\n\n")
    print("Output saved to output.txt")