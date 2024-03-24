class Arsenal_class():
    algorithm_implementation = Arsenal

    method_name = 'Arsenal'

    parameter_space = {
         'num_kernels': [1000, 2000, 3000],  # Number of kernels for each ROCKET transform.
        'n_estimators': [3, 5, 6],  # Number of estimators to build for the ensemble.
        'rocket_transform': ["rocket", "minirocket"],  # The type of Rocket transformer to use. #, "multirocket" # broken
        # Valid inputs = ["rocket", "minirocket", "multirocket"].
        'max_dilations_per_kernel': [16, 32, 64],  # MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
        'n_features_per_kernel': [3, 4, 5],  # MultiRocket only. The number of features per kernel.
        'time_limit_in_minutes': time_limit_param,  # Time contract to limit build time in minutes, overriding n_estimators. Default of 0 means n_estimators is used.
        'contract_max_n_estimators': [50, 100, 150],  # Max number of estimators when time_limit_in_minutes is set.
        #'save_transformed_data': [True, False],  # Save the data transformed in fit for use in _get_train_probs.
        'n_jobs': [n_jobs_model_val],  # The number of jobs to run in parallel for both fit and predict. -1 means using all processors.
        'random_state': [random_state_val],  # Seed for random number generation.
    }