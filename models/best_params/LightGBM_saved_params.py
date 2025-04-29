# LightGBM_saved_params.py

HYPERPARAMETERS = {
    'GV1': {'n_estimators': 2943, 'learning_rate': 0.00946413430815016, 'num_leaves': 2, 'max_depth': 12, 'colsample_bytree': 0.7336807525366973, 'reg_alpha': 16.041796800115723, 'reg_lambda': 0.46241476922960995, 'min_split_gain': 0.21967223808979622, 'subsample': 0.6593292117131304, 'linear_tree': False}, #Linear tree true performed best
    'GV3': {'n_estimators': 2169, 'learning_rate': 0.006203585866632918, 'num_leaves': 6, 'max_depth': 3, 'colsample_bytree': 0.5501709709501512, 'reg_alpha': 6.651970989209494e-06, 'reg_lambda': 67.12148677618323, 'min_split_gain': 3.3184849385432336e-05, 'subsample': 0.2002492183845991, 'min_child_samples': 475, 'linear_tree': False},
    'GV51': {'n_estimators': 2951, 'learning_rate': 0.03286202028083, 'num_leaves': 3, 'max_depth': 6, 'colsample_bytree': 0.8552996267089586, 'reg_alpha': 2.5137862899800223, 'reg_lambda': 3.9430716192876, 'min_split_gain': 0.6280661793078444, 'subsample': 0.49235297280952, 'linear_tree': False},
    'MB4': {'n_estimators': 2469, 'learning_rate': 0.006305086838580839, 'num_leaves': 24, 'max_depth': 12, 'colsample_bytree': 0.9169225803001793, 'reg_alpha': 0.0005033311938044666, 'reg_lambda': 0.004161142801434803, 'min_split_gain': 0.001347092605256956, 'subsample': 0.28568037942748326, 'min_child_samples': 101, 'linear_tree': False},
    'MB8': {'n_estimators': 2562, 'learning_rate': 0.0002721948710291453, 'num_leaves': 19, 'max_depth': 7, 'colsample_bytree': 0.3416680396301089, 'reg_alpha': 36.36662142829141, 'reg_lambda': 11.930641162327543, 'min_split_gain': 1.1485285846125317, 'subsample': 0.4455322004731267, 'min_child_samples': 20, 'linear_tree': False},
    'MB10': {'n_estimators': 2786, 'learning_rate': 0.001855360087115081, 'num_leaves': 2, 'max_depth': 8, 'colsample_bytree': 0.10760728276321369, 'reg_alpha': 4.855864348076413, 'reg_lambda': 0.000406429534719246, 'min_split_gain': 0.5764666575210088, 'subsample': 0.8535011524818334, 'min_child_samples': 491, 'linear_tree': False},
    'MB18': {'n_estimators': 2650, 'learning_rate': 0.07114476009343425, 'num_leaves': 11, 'max_depth': 5, 'colsample_bytree': 0.1450870712623418, 'reg_alpha': 26.61771692528141, 'reg_lambda': 0.1434715951720141, 'min_split_gain': 5.3994844097874335, 'subsample': 0.3991305878561679}
}

# HYPERPARAMETERS = {
#     'GV1': {}, #Linear tree true performed best
#     'GV3': {},
#     'GV51':	{},
#     'MB4': {},
#     'MB8': {},
#     'MB10': {},
#     'MB18': {}
# }