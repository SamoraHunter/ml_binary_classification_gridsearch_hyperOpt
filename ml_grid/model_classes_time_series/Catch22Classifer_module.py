


class Catch22Classifier_class():
    algorithm_implementation = Catch22Classifier

    method_name = 'Catch22Classifier'

    parameter_space = {
        'features': ["all", ["DN_HistogramMode_5", "DN_HistogramMode_10"], ...],  # List of catch22 features to extract
        'catch24': [True, False],                                                  # Extract mean, std, and 22 Catch22 features
        'outlier_norm': [True, False],                                             # Normalize during outlier Catch22 features
        'replace_nans': [True, False],                                             # Replace NaN/inf values from the transform
        'use_pycatch22': [True, False],                                            # Use C-based pycatch22 implementation
        'estimator': [RandomForestClassifier(n_estimators=200),                  # Sklearn estimator for building the model
                      DecisionTreeClassifier()],                              # Add more estimators if desired
        'random_state': [random_state_val],                                                # Random seed for random number generator
        'n_jobs': [n_jobs_model_val],                                                         # Number of jobs for parallel processing
       # 'parallel_backend': [None, "loky", "multiprocessing", "threading"],        # Parallelization backend options
    }