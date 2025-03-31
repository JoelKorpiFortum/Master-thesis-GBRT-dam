# LightGBM_params.py

HYPERPARAMETERS = {
    'GV1_extrapolation':  {
        'n_estimators': 2984, 
        'learning_rate': 0.018, 
        'max_depth': 11, 
        'num_leaves': 4, 
        'subsample': 0.84, 
        'feature_fraction': 0.57, 
        'min_gain_to_split': 6.7, 
        'reg_alpha': 0.0, 
        'reg_lambda': 1.56, 
        'boosting_type': 'gbdt', 
        'linear_tree': True
    },
    'GV3': {
        'n_estimators': 2133, 
        'learning_rate': 0.08, 
        'max_depth': 11, 
        'num_leaves': 27, 
        'subsample': 0.69, 
        'feature_fraction': 0.78, 
        'min_gain_to_split': 10.69, 
        'reg_alpha': 1.85, 
        'reg_lambda': 1.05, 
        'linear_tree': False
    },
    'GV51': {
        'n_estimators': 852, 
        'learning_rate': 0.08, 
        'max_depth': 7, 
        'num_leaves': 17, 
        'subsample': 0.66, 
        'feature_fraction': 0.78, 
        'min_gain_to_split': 2.04, 
        'reg_alpha': 2.89, 
        'reg_lambda': 0.7, 
        'linear_tree': False
    },
    'MB4': {
        'n_estimators': 1500, 
        'learning_rate': 0.099, 
        'max_depth': 3, 
        'num_leaves': 21, 
        'subsample': 0.86, 
        'feature_fraction': 0.29, 
        'min_gain_to_split': 5.35, 
        'reg_alpha': 2.58, 
        'reg_lambda': 4.6, 
        'boosting_type': 'gbdt', 
        'linear_tree': False},
    'MB8': {
        'n_estimators': 836, 
        'learning_rate': 0.078, 
        'max_depth': 5, 
        'num_leaves': 24, 
        'subsample': 0.83, 
        'feature_fraction': 0.42, 
        'min_gain_to_split': 3.26, 
        'reg_alpha': 0.01, 
        'reg_lambda': 2.15, 
        'boosting_type': 'dart', 
        'line-ar_tree': False},
    
    # 'MB10': {'n_estimators': 2033, 'learning_rate': 0.00502, 'max_depth': 8, 'num_leaves': 30, 'subsample': 0.82, 'feature_fraction': 0.4944319566255557, 'min_gain_to_split': 7.28, 'reg_alpha': 2.44, 'reg_lambda': 1.24, 'boosting_type': 'dart', 'linear_tree': False},
    
    # 'MB10': {'n_estimators': 814, 'learning_rate': 0.00201, 'max_depth': 12, 'num_leaves': 6, 'subsample': 0.72, 'feature_fraction': 0.29, 'min_gain_to_split': 5.83, 'reg_alpha': 9.85, 'reg_lambda': 7.07, 'boosting_type': 'dart', 'linear_tree': False},
    'MB10': {'n_estimators': 2081, 
             'learning_rate': 0.00101, 
             'max_depth': 9, 
             'num_leaves': 25, 
             'subsample': 1.0, 
             'feature_fraction': 0.45, 
             'min_gain_to_split': 12.8, 
             'reg_alpha': 1.86, 
             'reg_lambda': 4.65, 
             'boosting_type': 'dart', 
             'linear_tree': False
    },
    
    #  'n_estimators': 814, 'learning_rate': 0.00201, 'max_depth': 12, 'num_leaves': 6, 'subsample': 0.72, 'feature_fraction': 0.29000000000000004, 'min_gain_to_split': 5.83, 'reg_alpha': 9.85, 'reg_lambda': 7.07, 'boosting_type': 'dart', 'linear_tree': False

    'MB18': {
        'n_estimators': 613, 
        'learning_rate': 0.03801, 
        'max_depth': 10, 
        'num_leaves': 7, 
        'subsample': 0.99, 
        'feature_fraction': 0.48, 
        'min_gain_to_split': 0.04, 
        'reg_alpha': 6.53, 
        'reg_lambda': 2.77, 
        'linear_tree': False
    }
}