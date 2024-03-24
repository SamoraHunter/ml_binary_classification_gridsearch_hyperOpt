class TSFreshClassifier_class():
    algorithm_implementation = TSFreshClassifier

    method_name = 'TSFreshClassifier'

    parameter_space = {
        'default_fc_parameters': ['minimal', 'efficient', 'comprehensive'],   # Set of TSFresh features to be extracted
        'relevant_feature_extractor': [True, False],                          # Whether to remove irrelevant features using the FRESH algorithm
        'estimator': [None, RandomForestClassifier(n_estimators=200)],       # An sklearn estimator to be built using the transformed data
        'verbose': [verbose_param],                                                 # Level of output printed to the console
        'n_jobs': [n_jobs_model_val],                                                    # Number of jobs to run in parallel for fit and predict
        'chunksize': [None, 10, 100],                                         # Number of series processed in each parallel TSFresh job
        'random_state': [random_state_val],                                              # Seed for random number generation
    }