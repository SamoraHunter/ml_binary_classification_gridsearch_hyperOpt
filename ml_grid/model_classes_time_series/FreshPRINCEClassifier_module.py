from aeon.classification.feature_based._fresh_prince.FreshPRINCE import (
    FreshPRINCEClassifier,
)


class FreshPRINCEClassifier_class:

    def __init__(self, ml_grid_object):

        random_state_val = ml_grid_object.global_params.random_state_val

        verbose_param = ml_grid_object.verbose

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = FreshPRINCEClassifier

        self.method_name = "FreshPRINCEClassifier"

        self.parameter_space = {
            "default_fc_parameters": [
                "minimal",
                "efficient",
                "comprehensive",
            ],  # Set of TSFresh features to be extracted
            "n_estimators": [
                100,
                200,
                300,
            ],  # Number of estimators for the RotationForestClassifier ensemble
            "save_transformed_data": [False],  # Whether to save the transformed data
            "verbose": [
                verbose_param
            ],  # Level of output printed to the console (for information only)
            "n_jobs": [n_jobs_model_val],  # Number of jobs for parallel processing
            "chunksize": [
                None,
                100,
                200,
            ],  # Number of series processed in each parallel TSFresh job
            "random_state": [random_state_val],  # Seed for random, integer
        }
