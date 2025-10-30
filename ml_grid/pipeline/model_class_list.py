"""This module provides a function to get a list of model classes."""

import inspect
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from ml_grid.model_classes.adaboost_classifier_class import AdaBoostClassifierClass
from ml_grid.model_classes.catboost_classifier_class import (
    CatBoostClassifierClass,
)
from ml_grid.model_classes.gaussiannb_class import (
    GaussianNBClassifierClass,
)
from ml_grid.model_classes.gradientboosting_classifier_class import (
    GradientBoostingClassifierClass,
)
from ml_grid.model_classes.h2o_classifier_class import H2OAutoMLClass
from ml_grid.model_classes.h2o_deeplearning_classifier_class import (
    H2O_DeepLearning_class,
)
from ml_grid.model_classes.h2o_drf_classifier_class import H2ODRFClass as H2O_DRF_class # noqa
from ml_grid.model_classes.h2o_gam_classifier_class import H2OGAMClass as H2O_GAM_class
from ml_grid.model_classes.h2o_gbm_classifier_class import (
    H2O_GBM_class, # No change needed here
)  # Correctly named
from ml_grid.model_classes.h2o_glm_classifier_class import H2O_GLM_class
from ml_grid.model_classes.h2o_naive_bayes_classifier_class import (
    H2O_NaiveBayes_class,
)
from ml_grid.model_classes.h2o_rulefit_classifier_class import H2ORuleFitClass as H2O_RuleFit_class # noqa
from ml_grid.model_classes.h2o_stackedensemble_classifier_class import (
    H2O_StackedEnsemble_class,
)
from ml_grid.model_classes.h2o_xgboost_classifier_class import H2O_XGBoost_class
from ml_grid.model_classes.keras_classifier_class import KerasClassifierClass
from ml_grid.model_classes.knn_classifier_class import KNeighborsClassifierClass
from ml_grid.model_classes.knn_gpu_classifier_class import KNNGpuWrapperClass
from ml_grid.model_classes.light_gbm_class import LightGBMClassifierWrapper # noqa
from ml_grid.model_classes.logistic_regression_class import LogisticRegressionClass
from ml_grid.model_classes.mlp_classifier_class import MLPClassifierClass
from ml_grid.model_classes.NeuralNetworkClassifier_class import ( # noqa
    NeuralNetworkClassifier_class,
)
from ml_grid.model_classes.quadratic_discriminant_class import (
    QuadraticDiscriminantAnalysisClass,
)
from ml_grid.model_classes.randomforest_classifier_class import (
    RandomForestClassifierClass,
)
from ml_grid.model_classes.svc_class import SVCClass
from ml_grid.model_classes.tabtransformer_classifier_class import TabTransformerClass # noqa
from ml_grid.model_classes.xgb_classifier_class import XGBClassifierClass # noqa
from ml_grid.pipeline.data import pipe


def get_model_class_list(ml_grid_object: pipe) -> List[Any]:
    """Generates a list of instantiated model classes based on the configuration.

    This function reads the `model_class_dict` from the `ml_grid_object`. If the
    dictionary is not present, it uses a default dictionary. It then iterates
    through the dictionary, and for each model marked for inclusion, it
    instantiates the corresponding class using `eval()` and appends it to a list.

    Warning:
        The use of `eval()` can be dangerous if the input is not trusted. In this
        case, it is used to dynamically instantiate classes from their string
        names, which are defined within the project.

    Args:
        ml_grid_object (pipe): The main data pipeline object, which contains
            the training data and configuration.

    Returns:
        List[Any]: A list of instantiated model class objects.
    """
    logger = logging.getLogger("ml_grid")

    # Check if running in a CI environment (like GitHub Actions)
    is_ci_environment = os.environ.get("CI") == "true"

    # Check for GPU availability once
    gpu_available = torch.cuda.is_available()
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
            "LogisticRegressionClass": True,
            "KNeighborsClassifierClass": True,
            "QuadraticDiscriminantAnalysisClass": True,
            "SVCClass": True,
            "XGBClassifierClass": True,
            "MLPClassifierClass": True,
            "RandomForestClassifierClass": True,
            "GradientBoostingClassifierClass": True,
            "CatBoostClassifierClass": True,
            "GaussianNBClassifierClass": True,
            "LightGBMClassifierWrapper": True,
            "AdaBoostClassifierClass": True,
            "KerasClassifierClass": gpu_available,
            "KNNGpuWrapperClass": gpu_available,
            "NeuralNetworkClassifier_class": False,  # NNI based,
            "TabTransformerClass": False,  # PyTorch based
            "H2OAutoMLClass": False,  # H2O AutoML
            "H2O_GBM_class": True,  # H2O Gradient Boosting Machine
            "H2O_DRF_class": True,  # H2O Distributed Random Forest
            "H2O_DeepLearning_class": True,  # H2O Deep Learning
            "H2O_GLM_class": True,  # H2O Generalized Linear Model
            "H2O_NaiveBayes_class": True,  # H2O Naive Bayes
            "H2O_RuleFit_class": True,  # H2O RuleFit
            "H2O_XGBoost_class": True,  # H2O XGBoost
            "H2O_StackedEnsemble_class": True,  # H2O Stacked Ensemble
            "H2O_GAM_class": True,  # H2O Generalized Additive Models
        }

    # If running in a CI environment, explicitly disable resource-intensive models
    if is_ci_environment:
        logger.warning(
            "CI environment detected. Disabling GPU-heavy and resource-intensive models."
        )
        models_to_disable = [
            "KerasClassifierClass",
            "KNNGpuWrapperClass",
            "H2OAutoMLClass",
            "H2O_GBM_class",
            "H2O_DRF_class",
            "H2O_DeepLearning_class",
            "H2O_GLM_class",
            "H2O_NaiveBayes_class",
            "H2O_RuleFit_class",
            "H2O_XGBoost_class",
            "H2O_StackedEnsemble_class",
            "H2O_GAM_class",
            "TabTransformerClass",
        ]
        for model_name in models_to_disable:
            if model_name in model_class_dict:
                if model_class_dict[model_name]:
                    logger.info(f"Disabling '{model_name}' for CI run.")
                    model_class_dict[model_name] = False

    model_class_list = []

    for class_name, include in model_class_dict.items():
        if include:
            # Proactively skip GPU-specific models if no GPU is available
            if "_gpu_" in class_name.lower() and not gpu_available:
                logger.warning(
                    f"Skipping '{class_name}' because it requires a GPU, but no CUDA-enabled GPU is available."
                )
                continue
            # Try the exact name first, then try with '_class' appended for convenience
            try:
                model_class = eval(class_name)
            except NameError:
                class_name_with_suffix = f"{class_name}_class"
                try:
                    model_class = eval(class_name_with_suffix)
                except NameError:
                    raise NameError(
                        f"Could not find model class '{class_name}' or '{class_name_with_suffix}'. Please check the name and ensure it's imported."
                    )
            # Pass X and y to constructors that accept them (like H2OStackedEnsemble)
            init_signature = inspect.signature(model_class.__init__)
            init_params = {}
            if "X" in init_signature.parameters:
                init_params["X"] = ml_grid_object.X_train
            if "y" in init_signature.parameters:
                init_params["y"] = ml_grid_object.y_train
            if "parameter_space_size" in init_signature.parameters:
                init_params["parameter_space_size"] = parameter_space_size

            model_instance = model_class(**init_params)

            model_class_list.append(model_instance)

    return model_class_list
