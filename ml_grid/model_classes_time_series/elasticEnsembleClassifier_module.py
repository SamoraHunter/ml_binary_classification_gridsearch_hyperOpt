from aeon.classification.distance_based import ElasticEnsemble


class ElasticEnsemble_class:

    def __init__(self, ml_grid_object):

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = ElasticEnsemble()

        self.method_name = "ElasticEnsemble"

        self.parameter_space = {
            "proportion_of_param_options": [1.0, 0.8, 0.6],
            "proportion_train_in_param_finding": [1.0, 0.8, 0.6],
            "proportion_train_for_test": [1.0, 0.8, 0.6],
            "n_jobs": [n_jobs_model_val],
            "majority_vote": [False, True],
        }
