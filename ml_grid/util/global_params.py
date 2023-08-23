from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score



class global_parameters():
    
    
    def __init__(self, debug_level=0, knn_n_jobs=-1):
        
        
        
        self.debug_level = debug_level
        
        self.knn_n_jobs = knn_n_jobs
        
        self.verbose = 3
        
        self.rename_cols = True
        
        self.error_raise = False
        
        self.random_grid_search = True

        self.sub_sample_param_space_pct = 0.0005 #0.05==360 

        self.grid_n_jobs = 4

        self.metric_list = {'auc': make_scorer(roc_auc_score, needs_proba=False),
                        'f1':'f1',
                        'accuracy':'accuracy',
                        'recall': 'recall'}
        
        
    
    
    
    
  