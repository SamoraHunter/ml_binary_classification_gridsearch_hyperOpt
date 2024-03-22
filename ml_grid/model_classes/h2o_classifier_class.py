"""Define h2o_classifier_class"""

# +
from ml_grid.model_classes import H2OAutoMLClassifier
from ml_grid.util import param_space


# from h2o.sklearn import H2OAutoMLClassifier
# -


from ml_grid.model_classes.H2OAutoMLClassifier import *

print("Imported h2o_classifier_class")


class h2o_classifier_class:
    """h2o_classifier_class."""

    def __init__(self, X=None, y=None, parameter_space_size=None):
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        print("init h2o_classifier_class")
        self.X = X
        self.y = y

        self.algorithm_implementation = H2OAutoMLClassifier()
        self.method_name = "H2OAutoMLClassifier"

        self.parameter_vector_space = param_space.ParamSpace(parameter_space_size)
        # print(self.parameter_vector_space)

        self.parameter_space = [{"max_runtime_secs": [360], "nfolds": [2], "seed": [1]}]

        return None

        # print("init log reg class ", self.parameter_space)
