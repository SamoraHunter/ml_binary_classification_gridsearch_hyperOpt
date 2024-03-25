from aeon.classification.dictionary_based import ContractableBOSS


class ContractableBOSS_class:

    def __init__(self, ml_grid_object):

        time_limit_param = ml_grid_object.global_params.time_limit_param

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        random_state_val = ml_grid_object.global_params.random_state_val

        self.algorithm_implementation = ContractableBOSS

        self.method_name = "ContractableBOSS"

        self.parameter_space = {
            "n_parameter_samples": [100, 250, 500],  # Number of parameter combos to try
            "max_ensemble_size": [
                30,
                50,
                100,
            ],  # Maximum number of classifiers to retain
            "max_win_len_prop": [
                0.8,
                1.0,
            ],  # Maximum window length as a proportion of series length
            "min_window": [5, 10, 15],  # Minimum window size
            "time_limit_in_minutes": time_limit_param,  # Time contract to limit build time in minutes
            "contract_max_n_parameter_samples": [
                1000,
                2000,
            ],  # Max number of parameter combos when time_limit_in_minutes is set
            "save_train_predictions": [
                True,
                False,
            ],  # Save ensemble member train predictions in fit for LOOCV
            "n_jobs": [
                n_jobs_model_val
            ],  # Number of jobs to run in parallel for fit and predict
            "feature_selection": [
                "chi2",
                "none",
                "random",
            ],  # Sets the feature selection strategy to be used
            "random_state": [random_state_val],  # Seed for random integer
        }
