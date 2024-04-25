from aeon.classification.feature_based._signature_classifier import SignatureClassifier


class SignatureClassifier_class:

    def __init__(self, ml_grid_object):

        random_state_val = ml_grid_object.global_params.random_state_val

        self.algorithm_implementation = SignatureClassifier()

        self.method_name = "SignatureClassifier"

        self.parameter_space = {
            "random_state": [random_state_val],
        }
