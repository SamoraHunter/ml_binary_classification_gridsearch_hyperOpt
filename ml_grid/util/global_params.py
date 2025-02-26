import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score


    # Define your custom scoring function
def custom_roc_auc_score(y_true, y_pred):
    # Check if there are at least two unique classes present in y_true
    if len(np.unique(y_true)) < 2:
        return np.nan  # Return NaN if only one class is present
    else:
        return roc_auc_score(y_true, y_pred)
    
class GlobalParameters:
    """
    Global parameters for ml_grid

    Attributes:
        debug_level (int): Debug level, 0==minimal, 1,2,3,4
        knn_n_jobs (int): Number of jobs for knn, -1==all
        verbose (int): Verbose level for sklearn models
        rename_cols (bool): Rename cols of dataframes
        error_raise (bool): Raise errors from ml_grid
        random_grid_search (bool): Randomize search space for GridSearchCV
        sub_sample_param_space_pct (float): Percentage of param space to sub sample
        grid_n_jobs (int): Number of jobs for GridSearchCV
        time_limit_param (list): Time limit for GridSearchCV
        random_state_val (int): Random state value
        n_jobs_model_val (int): Number of jobs for models
        metric_list (dict): Dictionary of sklearn metrics to pass to GridSearchCV
        max_param_space_iter_value: hard limit on hyperparam search iterations.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GlobalParameters, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, debug_level=0, knn_n_jobs=-1):
        if self._initialized:
            return
        self._initialized = True

        self.debug_level = debug_level
        self.knn_n_jobs = knn_n_jobs
        self.verbose = 0
        self.rename_cols = True
        self.error_raise = False
        self.random_grid_search = False
        self.bayessearch = True
        self.sub_sample_param_space_pct = 0.0005  # 0.05==360
        self.grid_n_jobs = -1
        self.time_limit_param = [3]
        self.random_state_val = 1234
        self.n_jobs_model_val = -1
        self.max_param_space_iter_value = 10

        custom_scorer = make_scorer(custom_roc_auc_score)
        self.metric_list = {
            "auc": custom_scorer,
            "f1": "f1",
            "accuracy": "accuracy",
            "recall": "recall",
        }

    def update_parameters(self, **kwargs):
        """
        Update global parameters at runtime.

        Args:
            **kwargs: Key-value pairs of parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        """
        Dictionary of sklearn metrics to pass to GridSearchCV
        """
        custom_scorer = make_scorer(custom_roc_auc_score)



# Singleton instance
global_parameters = GlobalParameters()

    
    


