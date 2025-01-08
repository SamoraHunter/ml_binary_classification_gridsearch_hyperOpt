"""Define adaboost class"""

from ml_grid.util import param_space
from sklearn.naive_bayes import GaussianNB
from ml_grid.util.global_params import global_parameters
from scipy.stats import uniform
import numpy as np
from skopt.space import Categorical, Real

print("Imported gaussiannb class")

class GaussianNBWrapper(GaussianNB):
    
    
    def set_params(self, **params):
        prior_mapping = {
            0: None,  # Default priors (based on the class distribution in the dataset)
            1: [0.5, 0.5],  # Equal probabilities
            2: [0.6, 0.4],  # Slight imbalance favoring class 0
            3: [0.4, 0.6],  # Slight imbalance favoring class 1
            4: [0.7, 0.3],  # Moderate imbalance favoring class 0
            5: [0.3, 0.7],  # Moderate imbalance favoring class 1
            6: [0.8, 0.2],  # Strong imbalance favoring class 0
            7: [0.2, 0.8],  # Strong imbalance favoring class 1
            8: [0.9, 0.1],  # Extreme imbalance favoring class 0
            9: [0.1, 0.9],  # Extreme imbalance favoring class 1
            }
        
        if 'priors' in params:
            priors_idx = params.pop('priors')
            params['priors'] = prior_mapping[priors_idx]
        return super().set_params(**params)

class GaussianNB_class:
    """gaussiannb."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        
        global_params = global_parameters()
        self.X = X
        self.y = y

        if global_params.bayessearch is False:
            self.algorithm_implementation = GaussianNB()
        else:
            self.algorithm_implementation = GaussianNBWrapper() #Wrapper necessary for passing priors to bayescv
            
        self.method_name = "GaussianNB"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        # print(self.parameter_vector_space)

        if global_params.bayessearch:
            # For BayesSearchCV, use distributions from skopt.space

            self.parameter_space = {
                'var_smoothing': self.parameter_vector_space.param_dict.get("log_small"),
                'priors': Categorical([0, 1, 2])  # Integer mapping
            }

            # Log parameter space for verification
            print(f"Parameter Space: {self.parameter_space}")

        else:
            # For traditional grid search, use lists
            self.new_list = list(self.parameter_vector_space.param_dict.get("log_small")).copy()
            self.new_list.append(1e-9)  # Add the small value explicitly
            self.parameter_space = {
                "priors": [
                    None,
                    [0.1, 0.9],
                    [0.9, 0.1],
                    [0.7, 0.3],
                    [0.3, 0.7],
                    [0.5, 0.5],
                    [0.6, 0.4],
                    [0.4, 0.6],
                ],  # Enumerates possible values as a list
                "var_smoothing": self.new_list,
            }


        return None

        # print("init log reg class ", self.parameter_space)
