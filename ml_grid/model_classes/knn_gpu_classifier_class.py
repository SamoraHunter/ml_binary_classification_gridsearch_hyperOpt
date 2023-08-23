"""Define knn__gpu class"""


from ml_grid.model_classes.knn_wrapper_class import KNNWrapper
from ml_grid.util import param_space
from ml_grid.util.global_params import global_parameters

print("Imported knn__gpu class")






class knn__gpu_wrapper_class():
    """SVC."""
        
    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        self.X = X
        self.y = y

        self.algorithm_implementation = KNNWrapper()
        self.method_name = "knn__gpu"
        
        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        #print(self.parameter_vector_space)
        
        knn_n_jobs = global_parameters().knn_n_jobs

        self.parameter_space = {
           
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
#         "leaf_size": log_large_long,
        "metric": ["minkowski"],
        "metric_params": [None],
        #"n_jobs": [None],
        "n_jobs":[knn_n_jobs],
        "n_neighbors": self.parameter_vector_space.param_dict.get('log_med'),
        "p": self.parameter_vector_space.param_dict.get('log_med'),
#         "weights": ["uniform", "distance"],
        'device':['gpu'],
        'mode':['arrays','hdf5'],
        'scoring':["accuracy"]
          }
        
        
  
        
        return None
        
        
        

        #print("init log reg class ", self.parameter_space)
