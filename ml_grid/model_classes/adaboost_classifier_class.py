"""Define adaboost class"""

from ml_grid.util import param_space
from sklearn.ensemble import AdaBoostClassifier

print("Imported AdaBoostClassifier class")

class adaboost_class():
    """AdaBoostClassifier."""
        
    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = AdaBoostClassifier()
        self.method_name = "AdaBoostClassifier"
        
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        #print(self.parameter_vector_space)

        self.parameter_space = {
        "algorithm": ["SAMME.R", "SAMME"],
        "estimator": [None],
        "learning_rate": self.parameter_vector_space.param_dict.get('log_small'),
        "n_estimators": self.parameter_vector_space.param_dict.get('log_large_long'),
        "random_state": [None],
    }
        
        return None
        
        
        

        #print("init log reg class ", self.parameter_space)
