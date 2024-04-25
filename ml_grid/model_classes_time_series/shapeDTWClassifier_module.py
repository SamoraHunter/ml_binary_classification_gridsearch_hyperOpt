# univariate

ShapeDTW = None


class ShapeDTW_class:

    def __init__(self, ml_grid_object):

        n_jobs_model_val = ml_grid_object.global_params.n_jobs_model_val

        self.algorithm_implementation = ShapeDTW

        self.method_name = "ShapeDTW"

        self.parameter_space = {
            "n_neighbours": [-1],
            "subsequence_length": ["sqrt(n_timepoints)"],
            "shape_descriptor_function": ["raw"],
            "params": [None],
            "shape_descriptor_functions": [["raw", "derivative"]],
            "metric_params": [None],
            "n_jobs": [n_jobs_model_val],
        }
