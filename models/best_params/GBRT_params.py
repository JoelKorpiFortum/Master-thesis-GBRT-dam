# GBRT_params.py

HYPERPARAMETERS = {
    'GV1': {
        'max_iter': 914, 
        'learning_rate': 0.066, 
        'max_depth': 2, 
        'min_samples_leaf': 22, 
        'l2_regularization': 8.14, 
        'max_features': 0.77, 
        'early_stopping': 'auto'
    },
    'GV3': {
        'max_iter': 1747, 
        'learning_rate': 0.042, 
        'max_depth': 3, 
        'min_samples_leaf': 5, 
        'l2_regularization': 8.81, 
        'max_features': 0.89, 
        'early_stopping': 'auto'
    },
    'GV51': {
        'max_iter': 895, 
        'learning_rate': 0.037, 
        'max_depth': 2, 
        'min_samples_leaf': 29, 
        'l2_regularization': 3.39, 
        'max_features': 0.9, 
        'early_stopping': True
    },
    'MB4': {
        'max_iter': 2648,
        'learning_rate': 0.006,
        'max_depth': 7,
        'min_samples_leaf': 25,
        'l2_regularization': 6.75,
        'max_features': 0.65,
        'early_stopping': 'auto'
    },
    'MB8': {
        'max_iter': 1140,
        'learning_rate': 0.056,
        'max_depth': 2,
        'min_samples_leaf': 2,
        'l2_regularization': 5.81,
        'max_features': 0.78,
        'early_stopping': True
    },
    # 'MB10': {'max_iter': 807, 'learning_rate': 0.00901, 'max_depth': 7, 'min_samples_leaf': 26, 'l2_regularization': 7.23, 'max_features': 0.56, 'early_stopping': 'auto' }, # Proper fit
    'MB10': {'max_iter': 507, 
             'learning_rate': 0.00801, 
             'max_depth': 5, 
             'min_samples_leaf': 15, 
             'l2_regularization': 7.23, 
             'max_features': 0.56, 
             'early_stopping': 'auto' 
    }, # Proper fit
    'MB18': {
        'max_iter': 2340, 
        'learning_rate': 0.01701, 
        'max_depth': 3, 
        'min_samples_leaf': 14, 
        'l2_regularization': 1.64, 
        'max_features': 0.5, 
        'early_stopping': True
    }
}