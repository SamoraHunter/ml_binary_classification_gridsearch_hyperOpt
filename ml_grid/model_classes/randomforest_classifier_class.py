"""Define GradientBoostingClassifier class"""



from ml_grid.util import param_space
from sklearn.ensemble import RandomForestClassifier
import numpy as np

print("Imported RandomForestClassifier class")

class RandomForestClassifier_class():
    """RandomForestClassifier."""
        
    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = RandomForestClassifier()
        self.method_name = "RandomForestClassifier"
        
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        #print(self.parameter_vector_space)

        self.parameter_space = {
            
            "bootstrap": self.parameter_vector_space.param_dict.get('bool_param'),
            "ccp_alpha": self.parameter_vector_space.param_dict.get('lin_zero_one'),
            #'class_weight': [None, 'balanced'] + [{0: w} for w in [1, 2, 4, 6, 10]],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": self.parameter_vector_space.param_dict.get('log_med'),
            "max_features": ["sqrt", "log2"],
            # 'max_leaf_nodes': log_large_long,
            "max_samples": [None],
            #  'min_impurity_decrease': log_small,
            "min_samples_leaf": np.delete(self.parameter_vector_space.param_dict.get('log_med'), 0),
            "min_samples_split": np.delete(self.parameter_vector_space.param_dict.get('log_med'),0),
            #  'min_weight_fraction_leaf': log_small,
            "n_estimators": self.parameter_vector_space.param_dict.get('log_large_long'),
            "n_jobs": [None],
            "oob_score": [False],
            "random_state": [None],
            "verbose": [0],
            "warm_start": [False],
    }
        
        return None
        
        
        

        #print("init log reg class ", self.parameter_space)
