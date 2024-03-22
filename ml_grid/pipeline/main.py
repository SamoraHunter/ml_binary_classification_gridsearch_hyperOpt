import traceback

import ml_grid
import numpy as np
from ml_grid.model_classes.h2o_classifier_class import h2o_classifier_class
from ml_grid.model_classes.adaboost_classifier_class import adaboost_class
from ml_grid.model_classes.gaussiannb_class import GaussianNB_class
from ml_grid.model_classes.gradientboosting_classifier_class import (
    GradientBoostingClassifier_class,
)
from ml_grid.model_classes.keras_classifier_class import kerasClassifier_class
from ml_grid.model_classes.knn_classifier_class import knn_classifiers_class
from ml_grid.model_classes.knn_gpu_classifier_class import knn__gpu_wrapper_class
from ml_grid.model_classes.logistic_regression_class import LogisticRegression_class
from ml_grid.model_classes.mlp_classifier_class import mlp_classifier_class
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
from sklearn.model_selection import ParameterGrid
from ml_grid.model_classes.light_gbm_class import LightGBMClassifierWrapper
from ml_grid.model_classes.NeuralNetworkClassifier_class import (
    NeuralNetworkClassifier_class,
)

# from ml_grid.model_classes.H2OAutoMLClassifier import H2OAutoMLClassifier


class run:

    def __init__(self, ml_grid_object, local_param_dict):  # kwargs**

        self.global_params = global_parameters()

        self.verbose = self.global_params.verbose

        self.error_raise = self.global_params.error_raise

        self.ml_grid_object = ml_grid_object

        self.sub_sample_param_space_pct = self.global_params.sub_sample_param_space_pct

        self.parameter_space_size = local_param_dict.get("param_space_size")

        self.model_class_list = [
            #             NeuralNetworkClassifier_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train, # gpu error, memory overload on hyperopt
            #                          parameter_space_size=self.parameter_space_size),
            LogisticRegression_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            knn_classifiers_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            quadratic_discriminant_analysis_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            SVC_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            XGB_class_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            mlp_classifier_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            RandomForestClassifier_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            GradientBoostingClassifier_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            kerasClassifier_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            GaussianNB_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            adaboost_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            knn__gpu_wrapper_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            LightGBMClassifierWrapper(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
            h2o_classifier_class(
                X=self.ml_grid_object.X_train,
                y=self.ml_grid_object.y_train,
                parameter_space_size=self.parameter_space_size,
            ),
        ]

        if self.verbose >= 2:
            print(f"{len(self.model_class_list)} models loaded")

        self.pg_list = []

        for elem in self.model_class_list:

            pg = ParameterGrid(elem.parameter_space)

            self.pg_list.append(len(ParameterGrid(elem.parameter_space)))

            if self.verbose >= 1:
                print(f"{elem.method_name}:{len(pg)}")

            for param in elem.parameter_space:
                try:
                    if type(param) is not list:
                        if (
                            isinstance(elem.parameter_space.get(param), list) is False
                            and isinstance(elem.parameter_space.get(param), np.ndarray)
                            is False
                        ):
                            print("What is this?")
                            print(
                                f"{elem.method_name, param} {type(elem.parameter_space.get(param))}"
                            )

                except Exception as e:
                    #                     print(e)
                    pass

        # sample from mean of all param space n
        self.mean_parameter_space_val = np.mean(self.pg_list)

        self.sub_sample_parameter_val = int(
            self.sub_sample_param_space_pct * self.mean_parameter_space_val
        )

        # n_iter_v = int(sub_sample_param_space_pct *  len(ParameterGrid(parameter_space)))

        self.arg_list = []
        for model_class in self.model_class_list:

            class_name = model_class

            self.arg_list.append(
                (
                    class_name.algorithm_implementation,
                    class_name.parameter_space,
                    class_name.method_name,
                    self.ml_grid_object,
                    self.sub_sample_parameter_val,
                )
            )

        self.multiprocess = False

        self.local_param_dict = local_param_dict

        if self.verbose >= 2:
            print(f"Passed main init, len(arg_list): {len(self.arg_list)}")

    def execute(self):

        # needs implementing*

        self.model_error_list = []

        if self.multiprocess:

            def multi_run_wrapper(args):
                print("not implemented ")
                # return grid_search_cross_validate(*args)

            if __name__ == "__main__":
                from multiprocessing import Pool

                pool = Pool(8)
                results = pool.map(multi_run_wrapper, self.arg_list)
                # print(results)
                pool.close()  # exp

        elif self.multiprocess == False:
            for k in range(0, len(self.arg_list)):
                try:
                    print("grid searching...")
                    grid_search_cross_validate.grid_search_crossvalidate(
                        *self.arg_list[k]
                        # algorithm_implementation = LogisticRegression_class(parameter_space_size=self.parameter_space_size).algorithm_implementation, parameter_space = self.arg_list[k][1], method_name=self.arg_list[k][2], X = self.arg_list[k][3], y=self.arg_list[k][4]
                    )
                except Exception as e:

                    print(e)
                    print("error on ", self.arg_list[k][2])
                    self.model_error_list.append(
                        [self.arg_list[k][0], e, traceback.print_exc()]
                    )

                    if self.error_raise:
                        input(
                            "error thrown in grid_search_crossvalidate on model class list"
                        )

        print(
            f"Model error list: nb. errors returned from func: {self.model_error_list}"
        )
        print(self.model_error_list)

        return self.model_error_list
