"""Define KNeighborsClassifier class"""

from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters
from sklearn.neighbors import KNeighborsClassifier

print("Imported KNeighborsClassifier class")

class knn_classifiers_class():
    """KNeighborsClassifier."""
        
    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        knn_n_jobs = global_parameters().knn_n_jobs
        
        self.X = X
        self.y = y

        self.algorithm_implementation = KNeighborsClassifier()
        self.method_name = "KNeighborsClassifier"
        
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        #print(self.parameter_vector_space)

        self.parameter_space = {
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size":  self.parameter_vector_space.param_dict.get('log_large_long'),
        "metric": ["minkowski"],
        "metric_params": [None],
        #"n_jobs": [None],
        "n_jobs":[knn_n_jobs],
        "n_neighbors": self.parameter_vector_space.param_dict.get('log_med'),
        "p": self.parameter_vector_space.param_dict.get('log_med'),
        "weights": ["uniform", "distance"],
    }
        
        return None
        
        
        

        #print("init log reg class ", self.parameter_space)
