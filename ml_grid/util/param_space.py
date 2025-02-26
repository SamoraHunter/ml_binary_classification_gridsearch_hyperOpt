"""Import parameter space constants"""

import numpy as np
from skopt.space import Real, Integer, Categorical
# print("imported param_space")
from ml_grid.util.global_params import global_parameters

class ParamSpace:

    # print("param space called")

    def __init__(self, size):

        # print("param space init called")

        self.param_dict = None
        
        global_parameters_val = global_parameters
        
        if global_parameters_val is None:
            
            use_bayesian_param_space = False
        else:
            
            if global_parameters_val.bayessearch:
                
                use_bayesian_param_space = True
            else:
                
                use_bayesian_param_space = False
                
        if not use_bayesian_param_space:
                
            if size == "medium":

                nstep = 3
                self.param_dict = {
                    "log_epoch": np.floor(np.logspace(0, 1.5, 3)).astype(int),
                    "log_small": np.logspace(-1, -5, 3),
                    "bool_param": [True, False],
                    "log_large": np.logspace(0, 2, 3).astype(int),
                    "log_large_long": np.floor(np.logspace(0, 3.1, 5)).astype(int),
                    "log_med_long": np.floor(np.logspace(0, 1.5, 5)).astype(int),
                    "log_med": np.floor(np.logspace(0, 1.5, 3)).astype(int),
                    "log_zero_one": np.logspace(0.0, 1.0, nstep) / 10,
                    "lin_zero_one": np.linspace(0.0, 1.0, nstep) / 10,
                }

            if size == "xsmall":

                nstep = 2
                self.param_dict = {
                    "log_epoch": np.floor(np.logspace(-1, 1, 3)).astype(int),
                    "log_small": np.logspace(-1, -5, 2),
                    "bool_param": [True, False],
                    "log_large": np.logspace(0, 2, 2).astype(int),
                    "log_large_long": np.floor(np.logspace(0, 3.1, 2)).astype(int),
                    "log_med_long": np.floor(np.logspace(0, 1.5, 2)).astype(int),
                    "log_med": np.floor(np.logspace(0, 1.5, 2)).astype(int),
                    "log_zero_one": np.logspace(0.0, 1.0, nstep) / 10,
                    "lin_zero_one": np.linspace(0.0, 1.0, nstep) / 10,
                }

            if size == "xwide":

                nstep = 2
                self.param_dict = {
                    "log_epoch": np.floor(np.logspace(-2, 2, 3)).astype(int),
                    "log_small": np.logspace(-1, -5, 2),
                    "bool_param": [True, False],
                    "log_large": np.logspace(0, 2, 2).astype(int),
                    "log_large_long": np.floor(np.logspace(0, 3.1, 2)).astype(int),
                    "log_med_long": np.floor(np.logspace(0, 1.5, 2)).astype(int),
                    "log_med": np.floor(np.logspace(0, 1.5, 2)).astype(int),
                    "log_zero_one": np.logspace(0.0, 1.0, nstep) / 10,
                    "lin_zero_one": np.linspace(0.0, 1.0, nstep) / 10,
                }

        else:
    
        
            if size == "medium":
                nstep = 3
                self.param_dict = {
                    "log_epoch": Integer(np.floor(np.logspace(0, 1.5, 3)).min(), np.floor(np.logspace(0, 1.5, 3)).max()),
                    "log_small": Real(1e-5, 0.1, prior="log-uniform"),  # Correct usage of Real
                    "bool_param": Categorical([True, False]),  # Categorical for discrete values
                    "log_large": Integer(np.logspace(0, 2, 3).astype(int).min(), np.logspace(0, 2, 3).astype(int).max()),
                    "log_large_long": Integer(np.floor(np.logspace(0, 3.1, 5)).min(), np.floor(np.logspace(0, 3.1, 5)).max()),
                    "log_med_long": Integer(np.floor(np.logspace(0, 1.5, 5)).min(), np.floor(np.logspace(0, 1.5, 5)).max()),
                    "log_med": Integer(np.floor(np.logspace(0, 1.5, 3)).min(), np.floor(np.logspace(0, 1.5, 3)).max()),
                    "log_zero_one": Real(0.1, 1.0, prior="log-uniform"),  # Correct usage of Real
                    "lin_zero_one": Real(0.0, 0.1, prior="uniform"),  # Correct usage of Real
                }
            elif size == "xsmall":
                nstep = 2
                self.param_dict = {
                    "log_epoch": Integer(np.floor(np.logspace(-1, 1, 3)).min(), np.floor(np.logspace(-1, 1, 3)).max()),
                    "log_small": Real(1e-5, 0.1, prior="log-uniform"),  # Correct usage of Real
                    "bool_param": Categorical([True, False]),  # Categorical for discrete values
                    "log_large": Integer(np.logspace(0, 2, 2).astype(int).min(), np.logspace(0, 2, 2).astype(int).max()),
                    "log_large_long": Integer(np.floor(np.logspace(0, 3.1, 2)).min(), np.floor(np.logspace(0, 3.1, 2)).max()),
                    "log_med_long": Integer(np.floor(np.logspace(0, 1.5, 2)).min(), np.floor(np.logspace(0, 1.5, 2)).max()),
                    "log_med": Integer(np.floor(np.logspace(0, 1.5, 2)).min(), np.floor(np.logspace(0, 1.5, 2)).max()),
                    "log_zero_one": Real(0.1, 1.0, prior="log-uniform"),  # Correct usage of Real
                    "lin_zero_one": Real(0.0, 0.1, prior="uniform"),  # Correct usage of Real
                }
            elif size == "xwide":
                nstep = 2
                self.param_dict = {
                    "log_epoch": Integer(np.floor(np.logspace(-2, 2, 3)).min(), np.floor(np.logspace(-2, 2, 3)).max()),
                    "log_small": Real(1e-5, 0.1, prior="log-uniform"),  # Correct usage of Real
                    "bool_param": Categorical([True, False]),  # Categorical for discrete values
                    "log_large": Integer(np.logspace(0, 2, 2).astype(int).min(), np.logspace(0, 2, 2).astype(int).max()),
                    "log_large_long": Integer(np.floor(np.logspace(0, 3.1, 2)).min(), np.floor(np.logspace(0, 3.1, 2)).max()),
                    "log_med_long": Integer(np.floor(np.logspace(0, 1.5, 2)).min(), np.floor(np.logspace(0, 1.5, 2)).max()),
                    "log_med": Integer(np.floor(np.logspace(0, 1.5, 2)).min(), np.floor(np.logspace(0, 1.5, 2)).max()),
                    "log_zero_one": Real(0.01, 1.0, prior="log-uniform"),  # Correct usage of Real
                    "lin_zero_one": Real(0.0, 0.1, prior="uniform"),  # Correct usage of Real
                }



