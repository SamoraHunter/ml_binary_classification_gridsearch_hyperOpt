from typing import Any, Dict, List, Optional
import logging

from ml_grid.model_classes.adaboost_classifier_class import adaboost_class
from ml_grid.model_classes.catboost_classifier_class import CatBoost_class
from ml_grid.model_classes.gaussiannb_class import GaussianNB_class
from ml_grid.model_classes.gradientboosting_classifier_class import (
    GradientBoostingClassifier_class,
)
from ml_grid.model_classes.h2o_classifier_class import h2o_classifier_class
from ml_grid.model_classes.keras_classifier_class import kerasClassifier_class
from ml_grid.model_classes.knn_classifier_class import knn_classifiers_class
from ml_grid.model_classes.knn_gpu_classifier_class import knn__gpu_wrapper_class
from ml_grid.model_classes.light_gbm_class import LightGBMClassifierWrapper
from ml_grid.model_classes.logistic_regression_class import LogisticRegression_class
from ml_grid.model_classes.mlp_classifier_class import mlp_classifier_class
from ml_grid.model_classes.NeuralNetworkClassifier_class import (
    NeuralNetworkClassifier_class,
)
from ml_grid.model_classes.quadratic_discriminant_class import (
    quadratic_discriminant_analysis_class,
)
from ml_grid.model_classes.randomforest_classifier_class import (
    RandomForestClassifier_class,
)
from ml_grid.model_classes.svc_class import SVC_class
from ml_grid.model_classes.tabtransformer_classifier_class import TabTransformer_class
from ml_grid.model_classes.xgb_classifier_class import XGB_class_class

from ml_grid.pipeline.data import pipe


def get_model_class_list(ml_grid_object: pipe) -> List[Any]:
    """Generates a list of instantiated model classes based on the configuration.

    This function reads the `model_class_dict` from the `ml_grid_object`. If the
    dictionary is not present, it uses a default dictionary. It then iterates
    through the dictionary, and for each model marked for inclusion, it
    instantiates the corresponding class using `eval()` and appends it to a list.

    Note:
        The use of `eval()` is necessary for this function to work as intended,
        as it dynamically instantiates classes from their string names. All
        model classes must be imported into this module's scope.

    Args:
        ml_grid_object (pipe): The main data pipeline object, which contains
            the training data and configuration.

    Returns:
        List[Any]: A list of instantiated model class objects.
    """
    logger = logging.getLogger('ml_grid')
    # Get the parameter space size, defaulting to 'small' if not provided.
    # This prevents errors when the key is missing from the configuration.
    parameter_space_size = ml_grid_object.local_param_dict.get("param_space_size")
    if parameter_space_size is None:
        parameter_space_size = "small"

    model_class_dict: Optional[Dict[str, bool]] = ml_grid_object.model_class_dict

    if model_class_dict is None:
        if ml_grid_object.verbose >= 1:
            logger.info("model_class_dict is None, using default model_class_dict")

        model_class_dict = {
            "LogisticRegression_class": True,
            "knn_classifiers_class": True,
            "quadratic_discriminant_analysis_class": True,
            "SVC_class": True,
            "XGB_class_class": True,
            "mlp_classifier_class": True,
            "RandomForestClassifier_class": True,
            "GradientBoostingClassifier_class": True,
            "CatBoost_class": True,
            "GaussianNB_class": True,
            "LightGBMClassifierWrapper": True,
            "adaboost_class": True,
            "kerasClassifier_class": True,
            "knn__gpu_wrapper_class": True,
            "NeuralNetworkClassifier_class": False,
            "TabTransformer_class": False,
            "h2o_classifier_class": False,
        }

    model_class_list = []

    for class_name, include in model_class_dict.items():
        if include:
            # Try the exact name first, then try with '_class' appended for convenience
            try:
                model_class = eval(class_name)
            except NameError:
                class_name_with_suffix = f"{class_name}_class"
                try:
                    model_class = eval(class_name_with_suffix)
                except NameError:
                    raise NameError(f"Could not find model class '{class_name}' or '{class_name_with_suffix}'. Please check the name and ensure it's imported.")
            model_instance = model_class(
                X=ml_grid_object.X_train,
                y=ml_grid_object.y_train,
                parameter_space_size=parameter_space_size,
            )
            model_class_list.append(model_instance)

    return model_class_list
