"""Define xbg class"""


import numpy as np
import xgboost as xgb
from ml_grid.util import param_space

print("Imported xbg class")

class XGB_class_class():
    """xbg."""
        
    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = xgb.XGBClassifier()
        self.method_name = "xbg"
        
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        #print(self.parameter_vector_space)

        self.parameter_space = {
        "objective": ["binary:logistic"],
        #'use_label_encoder': bool_param,
        #'base_score': lin_zero_one,
        "booster": ["gbtree", "gblinear", "dart"],
        #'callbacks': [None],
        #'colsample_bylevel': None,
        #'colsample_bynode': None,
        #'colsample_bytree': None,
        #'early_stopping_rounds': [None],
        #'enable_categorical': bool_param,
        #'eval_metric': [None],
        "gamma": self.parameter_vector_space.param_dict.get('log_small'),
        #'gpu_id': None,
        "grow_policy": [0, 1],
        "grow_policy": ['depthwise', 'lossguide'],
        #'importance_type': None,
        #'interaction_constraints': None,
        "learning_rate": self.parameter_vector_space.param_dict.get('log_small'),
        'max_bin': self.parameter_vector_space.param_dict.get('log_large_long'),
        #'max_cat_to_onehot': None,
        #'max_delta_step': None,
        'max_depth': self.parameter_vector_space.param_dict.get('log_large_long'),
        'max_leaves': self.parameter_vector_space.param_dict.get('log_large_long'),
        'min_child_weight': [None],
        "missing": [np.nan],
        #'monotone_constraints': None,
        "n_estimators": self.parameter_vector_space.param_dict.get('log_large_long'),
        "n_jobs": [-1],
        #'num_parallel_tree': None,
        #'predictor': None,
        "random_state": [None],
        "reg_alpha": self.parameter_vector_space.param_dict.get('log_small'),
        "reg_lambda": self.parameter_vector_space.param_dict.get('log_small'),
        "sampling_method": ["uniform", "gradient_based"],
        #'scale_pos_weight': None,
        #'subsample': None,
        # "tree_method": ["gpu_hist"], #fail w/ hyperopts?
        #'validate_parameters': None,
        "verbosity": [0],
        
        }
        
        return None
        
        
        

        #print("init log reg class ", self.parameter_space)
