from sklearn.model_selection import ParameterGrid

from ml_grid.model_classes.adaboost_classifier_class import adaboost_class
from ml_grid.model_classes.catboost_classifier_class import CatBoost_class
from ml_grid.model_classes.gaussiannb_class import GaussianNB_class
from ml_grid.model_classes.gradientboosting_classifier_class import (
    GradientBoostingClassifier_class,
)
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

# from ml_grid.model_classes import LogisticRegression_class
from ml_grid.pipeline import grid_search_cross_validate
from ml_grid.util import grid_param_space
from ml_grid.util.global_params import global_parameters


def get_model_class_list(ml_grid_object):
    parameter_space_size = ml_grid_object.local_param_dict.get("param_space_size")
    model_class_dict = ml_grid_object.model_class_dict
    if(model_class_dict == None and ml_grid_object.verbose >=1):
        print("model_class_dict is None, using default model_class_dict")

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
        "NeuralNetworkClassifier_class": True,
        "TabTransformer_class": False,
    }

    model_class_list = []

    for class_name, include in model_class_dict.items():
        if include:
            model_class = eval(class_name)
            model_instance = model_class(X=ml_grid_object.X_train,
                                         y=ml_grid_object.y_train,
                                         parameter_space_size=parameter_space_size)
            model_class_list.append(model_instance)

    return model_class_list

