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
    
class global_parameters:
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
    """

    def __init__(self, debug_level=0, knn_n_jobs=-1):
        """
        Initialize global parameters

        Args:
            debug_level (int): Debug level, 0==minimal, 1,2,3,4
            knn_n_jobs (int): Number of jobs for knn, -1==all
        """
        self.debug_level = debug_level
        self.knn_n_jobs = knn_n_jobs

        """
        Verbose level for sklearn models
        """
        self.verbose = 9

        """
        Rename cols of dataframes
        """
        self.rename_cols = True

        """
        Raise errors from ml_grid
        """
        self.error_raise = False

        """
        Randomize search space for GridSearchCV
        """
        self.random_grid_search = True

        """
        Percentage of param space to sub sample
        """
        self.sub_sample_param_space_pct = 0.0005  # 0.05==360

        """
        Number of jobs for GridSearchCV
        """
        self.grid_n_jobs = 4

        """
        Time limit for GridSearchCV
        """
        self.time_limit_param = [3]

        """
        Random state value
        """
        self.random_state_val = 0

        """
        Number of jobs for models
        """
        self.n_jobs_model_val = 2

        """
        Dictionary of sklearn metrics to pass to GridSearchCV
        """
        custom_scorer = make_scorer(custom_roc_auc_score)

        self.metric_list = {
            #"auc": make_scorer(roc_auc_score, needs_proba=False),
            #"auc": "roc_auc",
            "auc": custom_scorer,
            "f1": "f1",
            "accuracy": "accuracy",
            "recall": "recall",
        }



    
    


