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
from ml_grid.model_classes.xgb_classifier_class import XGB_class_class

# from ml_grid.model_classes import LogisticRegression_class
from ml_grid.pipeline import grid_search_cross_validate
from ml_grid.util import grid_param_space
from ml_grid.util.global_params import global_parameters


def get_model_class_list(ml_grid_object):

    parameter_space_size = ml_grid_object.local_param_dict.get("param_space_size")
    model_class_list = [
        #             NeuralNetworkClassifier_class(X=ml_grid_object.X_train, y=ml_grid_object.y_train, # gpu error, memory overload on hyperopt
        #                          parameter_space_size=parameter_space_size),
        LogisticRegression_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        knn_classifiers_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        quadratic_discriminant_analysis_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        SVC_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        XGB_class_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        mlp_classifier_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        RandomForestClassifier_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        GradientBoostingClassifier_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        kerasClassifier_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        GaussianNB_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        adaboost_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        knn__gpu_wrapper_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        LightGBMClassifierWrapper(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
        CatBoost_class(
            X=ml_grid_object.X_train,
            y=ml_grid_object.y_train,
            parameter_space_size=parameter_space_size,
        ),
    ]
    return model_class_list
