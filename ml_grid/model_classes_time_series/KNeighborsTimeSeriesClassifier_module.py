from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier


class KNeighborsTimeSeriesClassifier_class:

    def __init__(self, ml_grid_object):

        self.algorithm_implementation = KNeighborsTimeSeriesClassifier

        self.method_name = "KNeighborsTimeSeriesClassifier"

        self.parameter_space = {
            "distance": [
                "dtw",
                "euclidean",
            ],  # , 'cityblock' 'ctw', 'sqeuclidean','sax' 'softdtw'
            "n_neighbors": [2, 3, 5],  # [log_med_long]
            "n_jobs": [ml_grid_object.n_jobs_model_val],
        }

        # nb consider probability scoring on binary class eval: CalibratedClassifierCV

        return None
