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
    # 'max_iter': 2228, 'learning_rate': 0.028642509665772836, 'max_depth': 2, 'min_samples_leaf': 6, 'l2_regularization': 6.065107283139244, 'max_features': 0.8360692195778795, 'early_stopping': 'auto'}
    'GV3': {
        'max_iter': 1747, 
        'learning_rate': 0.042, 
        'max_depth': 3, 
        'min_samples_leaf': 5, 
        'l2_regularization': 8.81, 
        'max_features': 0.89, 
        'early_stopping': 'auto'
    },
    # {'max_iter': 1573, 'learning_rate': 0.05611533187208159, 'max_depth': 10, 'min_samples_leaf': 16, 'l2_regularization': 1.2457594918651602e-05, 'max_features': 0.6798092487168154, 'early_stopping': True}
    'GV51': {
        'max_iter': 895, 
        'learning_rate': 0.037, 
        'max_depth': 2, 
        'min_samples_leaf': 29, 
        'l2_regularization': 3.39, 
        'max_features': 0.9, 
        'early_stopping': True
    },
    # {'max_iter': 2482, 'learning_rate': 0.009176097866463569, 'max_depth': 2, 'min_samples_leaf': 7, 'l2_regularization': 2.5007204728032145e-07, 'max_features': 0.4789848045894612, 'early_stopping': 'auto'}
    'MB4': {
        'max_iter': 2648,
        'learning_rate': 0.006,
        'max_depth': 7,
        'min_samples_leaf': 25,
        'l2_regularization': 6.75,
        'max_features': 0.65,
        'early_stopping': 'auto'
    },
    # {'max_iter': 2568, 'learning_rate': 0.052484943912190274, 'max_depth': 8, 'min_samples_leaf': 10, 'l2_regularization': 5.834616619686079, 'max_features': 0.6211770931069401, 'early_stopping': True}
    'MB8': {'max_iter': 1158, 'learning_rate': 0.0024014123075105373, 'max_depth': 6, 'min_samples_leaf': 8, 'l2_regularization': 9.273107706273734, 'max_features': 0.13952678672446914, 'early_stopping': 'auto'},
    # {'max_iter': 1623, 'learning_rate': 0.0014285771830109765, 'max_depth': 7, 'min_samples_leaf': 9, 'l2_regularization': 0.09046229572054644, 'max_features': 0.10066033804866016, 'early_stopping': 'auto'},
    'MB10': {'max_iter': 507, 
             'learning_rate': 0.00801, 
             'max_depth': 5, 
             'min_samples_leaf': 15, 
             'l2_regularization': 7.23, 
             'max_features': 0.56, 
             'early_stopping': 'auto' 
    },
    # 'max_iter': 1509, 'learning_rate': 0.07114476009343425, 'max_depth': 10, 'min_samples_leaf': 16, 'l2_regularization': 3.6323392569431376e-07, 'max_features': 0.14321698289111517, 'early_stopping': 'auto'}
    'MB18': {
        'max_iter': 2340, 
        'learning_rate': 0.01701, 
        'max_depth': 3, 
        'min_samples_leaf': 14, 
        'l2_regularization': 1.64, 
        'max_features': 0.5, 
        'early_stopping': True
    }
    # 	{'max_iter': 1821, 'learning_rate': 0.041842331974188185, 'max_depth': 4, 'min_samples_leaf': 4, 'l2_regularization': 0.30797671509923463, 'max_features': 0.12182469588616546, 'early_stopping': 'auto'}
}