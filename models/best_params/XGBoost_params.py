# XGBoost_params.py

HYPERPARAMETERS = {
    'GV1': {
        'n_estimators': 820, 
        'learning_rate': 0.00901, 
        'max_depth': 9, 
        'max_leaves': 6, 
        'colsample_bytree': 0.68, 
        'subsample': 0.88, 
        'reg_alpha': 3.97, 
        'reg_lambda': 3.34, 
        'gamma': 4.07
    },
    'GV3': {
        'n_estimators': 2079, 
        'learning_rate': 0.00301, 
        'max_depth': 10, 
        'max_leaves': 11, 
        'colsample_bytree': 0.57, 
        'subsample': 0.64, 
        'reg_alpha': 8.91, 
        'reg_lambda': 7.29, 
        'gamma': 3.99
    },
    'GV51': {
        'n_estimators': 1497, 
        'learning_rate': 0.05401, 
        'max_depth': 7, 
        'max_leaves': 4, 
        'colsample_bytree': 0.74, 
        'subsample': 0.87, 
        'reg_alpha': 4.64, 
        'reg_lambda': 8.38, 
        'gamma': 0.56
    },
    'MB4': {
        'n_estimators': 1829, 
        'learning_rate': 0.05201, 
        'max_depth': 5, 
        'max_leaves': 10, 
        'colsample_bytree': 0.71, 
        'subsample': 0.91, 
        'reg_alpha': 6.52, 
        'reg_lambda': 0.24, 
        'gamma': 0.78
    },
    'MB8': {
        'n_estimators': 814, 
        'learning_rate': 0.00101, 
        'max_depth': 10, 
        'max_leaves': 28, 
        'colsample_bytree': 0.73, 
        'subsample': 0.678, 
        'reg_alpha': 1.99, 
        'reg_lambda': 4.27, 
        'gamma': 2.28
    },
    'MB10': {
        'n_estimators': 2805, 
        'learning_rate': 0.01401, 
        'max_depth': 9, 
        'max_leaves': 17, 
        'colsample_bytree': 0.5, 
        'subsample': 0.96, 
        'reg_alpha': 0.05, 
        'reg_lambda': 0.14, 
        'gamma': 1.39
    }, # Optimal Crossval
    'MB18': {
        'n_estimators': 2001, 
        'learning_rate': 0.01301, 
        'max_depth': 9, 
        'max_leaves': 8, 
        'colsample_bytree': 0.65, 
        'subsample': 0.91, 
        'reg_alpha': 0.12, 
        'reg_lambda': 2.5, 
        'gamma': 4.68
    }
}