# data_preparation.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FILE_PATHS = {
    'ÖVY': './data/SVG_ÖVY.csv',
    'NVY': './data/SVG_NVY.csv',
    'MÄTB_2': './data/SVG_MÄTB_2.csv',
    'MÄTB_6': './data/SVG_MÄTB_6.csv',
    'DPM_S0705BL01': './data/SVG_DPM_S0705BL01.csv',
    'DPM_S0705BL01_anomaly': './data/SVG_DPM_S0705BL01_anomaly.csv',
    'DPM_S0710BL14': './data/SVG_DPM_S0710BL14.csv',
    'DPM_S0712BL16': './data/SVG_DPM_S0712BL16.csv',
    'DPM_S0713BL17': './data/SVG_DPM_S0713BL17.csv',
    'DPM_S0714BL18': './data/SVG_DPM_S0714BL18.csv',
    'TOTAL': './data/SVG_TOTAL.csv',
    'VATTENTEMP': './data/SVG_VATTENTEMP.csv',
    'PRECIP': './data/SVGA_PRECIP.csv',
}

DATASET_MAPPING = {
    'h': 'ÖVY',
    'NVY':'NVY',
    'MÄTB_2': 'MÄTB_2',
    'MÄTB_6': 'MÄTB_6',
    'MÄTB_2_lag_1': 'derived',
    'MÄTB_2_lag_2': 'derived',
    'DPM_05':'DPM_S0705BL01',
    'DPM_05_anomaly':'DPM_S0705BL01_anomaly',
    'DPM_05_lag_1': 'derived',
    'DPM_05_anomaly_lag_1':'derived',
    'DPM_05_lag_2': 'derived',
    'DPM_10':'DPM_S0710BL14',
    'DPM_10_lag_1': 'derived',
    'DPM_12':'DPM_S0712BL16',
    'DPM_13':'DPM_S0713BL17',
    'DPM_14':'DPM_S0714BL18',
    'TOTAL':'TOTAL',
    'T_W':'VATTENTEMP',
    'P':'PRECIP',
    'h_poly': 'derived',
    'Sin_s': 'derived', 
    'Cos_s': 'derived',  
    'Sin_2s': 'derived',     
    'Cos_2s': 'derived',     
    't': 'derived',       
    'ln_t': 'derived', 
    # 'feature_name': 'dataset_key',
}

DERIVED_FEATURE_BASES = {
    'MÄTB_2_lag_1': 'MÄTB_2',
    'MÄTB_2_lag_2': 'MÄTB_2',
    'DPM_05_lag_1': 'DPM_05',
    'DPM_05_lag_2': 'DPM_05',
    'DPM_05_anomaly_lag_1':'DPM_05_anomaly',
    'DPM_10_lag_1': 'DPM_10',
    'h_poly': 'h',
}

def load_and_split_dataset(path, common_start_date=None, common_end_date=None, test_size=0.2):
    df = pd.read_csv(path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df = df.groupby(df.index).mean()  # Aggregate duplicates (winter time change)

    if common_start_date:
        start_date = pd.to_datetime(common_start_date)
    else:
        start_date = df.index.min()
    
    if common_end_date:
        end_date = pd.to_datetime(common_end_date)
    else:
        end_date = df.index.max()
        
    all_dates = pd.date_range(start=start_date, end=end_date, freq='h')
    df_reindexed = df.reindex(all_dates)

    # Split into train and test 
    split_idx = int(len(df_reindexed) * (1 - test_size))
    train = df_reindexed.iloc[:split_idx]
    test = df_reindexed.iloc[split_idx:]

    # Interpolate separately on train and test sets
    train_interpolated = train.interpolate(method='time')
    test_interpolated = test.interpolate(method='time')

    # Combine interpolated sets
    df_interpolated = pd.concat([train_interpolated, test_interpolated])
    df_interpolated.reset_index(inplace=True)
    df_interpolated.rename(columns={'index': 'Date'}, inplace=True)
    
    return df_interpolated

def load_and_merge_data(features, target, start_date, end_date, test_size=0.2):
    datasets_required = set()
    for variable in features + [target]:
        if variable in DATASET_MAPPING:
            dataset = DATASET_MAPPING[variable]
            if dataset != 'derived':
                datasets_required.add(dataset)
            else:
                base_variable = DERIVED_FEATURE_BASES.get(variable)
                if base_variable:
                    base_dataset = DATASET_MAPPING.get(base_variable)
                    if base_dataset and base_dataset != 'derived':
                        datasets_required.add(base_dataset)
        else:
            raise ValueError(f"Variable '{variable}' not found in DATASET_MAPPING.")

    loaded_dataframes = {}
    for dataset_key in datasets_required:
        path = FILE_PATHS.get(dataset_key)
        if path is None:
            raise ValueError(f"Dataset '{dataset_key}' not found in FILE_PATHS.")
        df_interpolated = load_and_split_dataset(path, common_start_date=start_date, common_end_date=end_date, test_size=test_size)
        loaded_dataframes[dataset_key] = df_interpolated

    df_merged = pd.DataFrame()
    for df in loaded_dataframes.values():
        if df_merged.empty:
            df_merged = df
        else:
            df_merged = pd.merge(df_merged, df, on='Date', how='inner')

    return df_merged

def preprocess_data(features, target, start_date, end_date, poly_degree=None, test_size=0.2):
    df = load_and_merge_data(features, target, start_date, end_date, test_size=test_size)
    df_preprocessed = pd.DataFrame()
    df_preprocessed = df[['Date', target]].copy()

    if any(feature in features for feature in ['Sin_s', 'Cos_s', 'Sin_2s', 'Cos_2s']):
        day_of_year = df['Date'].dt.dayofyear
        s = 2 * np.pi * day_of_year / 365.0
        if 'Sin_s' in features:
            df_preprocessed['Sin_s'] = np.sin(s)
        if 'Cos_s' in features:
            df_preprocessed['Cos_s'] = np.cos(s)
        if 'Sin_2s' in features:
            df_preprocessed['Sin_2s'] = np.sin(2 * s)
        if 'Cos_2s' in features:
            df_preprocessed['Cos_2s'] = np.cos(2 * s)
    
    if 't' in features or 'ln_t' in features:
        min_date = df['Date'].min()
        df_preprocessed['t'] = (df['Date'] - min_date).dt.days + 1
        if 'ln_t' in features:
            df_preprocessed['ln_t'] = np.log(df_preprocessed['t'])

    for feature in features:
        if DATASET_MAPPING.get(feature) == 'derived':
            if '_lag_' in feature:
                base_var, lag_num = feature.split('_lag_')
                lag_num = int(lag_num)
                df_preprocessed[feature] = df[base_var].shift(lag_num)
            elif feature == 'h_poly' and poly_degree is not None:
                poly = PolynomialFeatures(degree=poly_degree)
                h_poly = poly.fit_transform(df[['h']])
                for i in range(1, poly_degree + 1):
                    df_preprocessed[f'h_poly_{i}'] = h_poly[:, i]
        elif feature in df:
            df_preprocessed[feature] = df[feature]

    df_preprocessed.dropna(inplace=True)
    dates = df_preprocessed['Date']
    y = df_preprocessed[target]
    X = df_preprocessed.drop(columns=['Date', target])

    return X, y, dates

def split_data(X, y, test_size=0.3, split_index_from_end=None):
    if split_index_from_end:
        split_idx = len(y) - split_index_from_end
    else:
        split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx] if y is not None else None
    y_test = y.iloc[split_idx:] if y is not None else None

    # Normalize features on train set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    # Apply the same scaler to test set
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, split_idx

# # Test
# if __name__ == "__main__":
#     features = ['h_poly', 'DPM_10', 'DPM_12', 'DPM_13', 'DPM_14', 'NVY', 'TOTAL', 'P', 'T_W','t', 'ln_t', 'Cos_s', 'Sin_s', 'Cos_2s', 'Sin_2s']
#     poly_degree = 4
#     target = 'DPM_05'
#     start_date = "01-01-2019" #2020-07-09T17:00:00
#     end_date = "03-01-2024" ##2025-03-03T17:00:00
#     test_size = 0.3
    
#     X, y, _, _= preprocess_data(features, target, start_date, end_date, poly_degree=poly_degree, test_size=test_size)
#     X_train, X_test, y_train, y_test, split_idx = split_data(X, y, test_size=test_size)
#     print("Data preparation completed.")
