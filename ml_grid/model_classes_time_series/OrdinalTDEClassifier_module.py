from aeon.classification.ordinal_classification import OrdinalTDE


# unknown ts
class OrdinalTDE_class:

    def __init__(self, ml_grid_object):

        random_state_val = ml_grid_object.global_params.random_state_val

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        time_limit_param = ml_grid_object.global_params.time_limit_param

        self.algorithm_implementation = OrdinalTDE

        self.method_name = "OrdinalTDE"

        self.parameter_space = {
            "n_parameter_samples": [
                100,
                250,
                500,
            ],  # Number of parameter combinations to consider for the final ensemble
            "max_ensemble_size": [
                30,
                50,
                100,
            ],  # Maximum number of estimators in the ensemble
            "max_win_len_prop": [
                0.8,
                1.0,
            ],  # Maximum window length as a proportion of series length
            "min_window": [5, 10, 15],  # Minimum window length
            "randomly_selected_params": [
                30,
                50,
                70,
            ],  # Number of parameters randomly selected before Gaussian process parameter selection
            "bigrams": [True, False, None],  # Whether to use bigrams
            "dim_threshold": [
                0.75,
                0.85,
                0.95,
            ],  # Dimension accuracy threshold for multivariate data
            "max_dims": [
                10,
                20,
                30,
            ],  # Max number of dimensions per classifier for multivariate data
            "time_limit_in_minutes": [
                time_limit_param
            ],  # Time contract to limit build time in minutes
            "contract_max_n_parameter_samples": [
                1000,
                2000,
            ],  # Max number of parameter combinations when time_limit_in_minutes is set
            "typed_dict": [
                True,
                False,
            ],  # Whether to use numba typed Dict to store word counts
            #'save_train_predictions': [True, False],               # Save ensemble member train predictions in fit for LOOCV
            "n_jobs": [
                n_jobs_model_val
            ],  # Number of jobs to run in parallel for fit and predict
            "random_state": [random_state_val],  # Seed for random number generation
        }
