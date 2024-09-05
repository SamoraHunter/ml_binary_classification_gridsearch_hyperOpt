"""Define SVC class"""

from ml_grid.util import param_space
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print("Imported SVC class")


class SVC_class:
    """SVC."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        self.X = X
        self.y = y

        # Enforce scaling for SVM method
        if not self.is_data_scaled():
            self.scale_data()

        self.algorithm_implementation = SVC()
        self.method_name = "SVC"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        # print(self.parameter_vector_space)

        self.parameter_space = {
            "C": self.parameter_vector_space.param_dict.get("log_small"),
            "break_ties": self.parameter_vector_space.param_dict.get("bool_param"),
            #'cache_size': [200],
            #'class_weight': [None, 'balanced'] + [{0: w} for w in [1, 2, 4, 6, 10]], # enumerate class weight
            "coef0": self.parameter_vector_space.param_dict.get("log_small"),
            "decision_function_shape": ["ovr"],  # , 'ovo'
            "degree": self.parameter_vector_space.param_dict.get("log_med"),
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "max_iter": self.parameter_vector_space.param_dict.get("log_large_long"),
            #'probability': [False],
            #'random_state': [None],
            "shrinking": self.parameter_vector_space.param_dict.get("bool_param"),
            "tol": self.parameter_vector_space.param_dict.get("log_small"),
            "verbose": [False],
        }

        return None

        # print("init log reg class ", self.parameter_space)

    def is_data_scaled(self):
        """
        Check if data has been scaled to [0, 1] or [-1, 1] range. 
        
        This function calculates the minimum and maximum values for each feature in the data, 
        and checks if they are within the expected range for a scaled dataset. If they are, 
        the function returns True, indicating that the data has been scaled. If not, 
        the function returns False.
        
        The expected range for a scaled dataset is either [0, 1] or [-1, 1] for each feature. 
        If the minimum value is less than 0 or the maximum value is greater than 1, or 
        if the minimum value is less than -1 or the maximum value is greater than 1, 
        then the data has not been scaled.
        
        Returns:
            bool: True if data has been scaled, False if not.
        """
        
        # Calculate the range of values for each feature
        min_val = self.X.min().min()
        max_val = self.X.max().max()
        
        # Check if data is scaled to [0, 1] or [-1, 1] range
        if (min_val >= 0 and max_val <= 1) or (min_val >= -1 and max_val <= 1):
            return True
        else:
            return False

    
    def scale_data(self):
        # Initialize MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit and transform the data
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)
    
    