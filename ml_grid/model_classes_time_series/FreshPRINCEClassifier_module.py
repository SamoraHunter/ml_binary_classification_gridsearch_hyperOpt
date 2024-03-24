class FreshPRINCEClassifier_class():
    algorithm_implementation = FreshPRINCEClassifier

    method_name = 'FreshPRINCEClassifier'

    parameter_space = {
        'default_fc_parameters': ['minimal', 'efficient', 'comprehensive'],  # Set of TSFresh features to be extracted
        'n_estimators': [100, 200, 300],                 # Number of estimators for the RotationForestClassifier ensemble
        'save_transformed_data': [False],          # Whether to save the transformed data
        'verbose': [verbose_param],                            # Level of output printed to the console (for information only)
        'n_jobs': [n_jobs_model_val],                              # Number of jobs for parallel processing
        'chunksize': [None, 100, 200],                   # Number of series processed in each parallel TSFresh job
        'random_state': [random_state_val],                 # Seed for random, integer
    }