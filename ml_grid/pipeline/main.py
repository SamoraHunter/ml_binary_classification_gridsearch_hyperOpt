import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
from catboost import CatBoostError
from ml_grid.pipeline import grid_search_cross_validate
from ml_grid.pipeline.data import pipe
from ml_grid.util.bayes_utils import calculate_combinations
from ml_grid.util.global_params import global_parameters
from sklearn.model_selection import ParameterGrid

class run:
    """Orchestrates the hyperparameter search for a list of models."""

    def __init__(self, ml_grid_object: pipe, local_param_dict: Dict[str, Any]):
        """Initializes the run class.

        This class takes the main data pipeline object and a dictionary of local
        parameters to set up and prepare for executing a series of hyperparameter
        searches across multiple machine learning models.

        Args:
            ml_grid_object (pipe): The main data pipeline object, which contains
                the data (X_train, y_train, etc.) and a list of model classes
                to be evaluated.
            local_param_dict (Dict[str, Any]): A dictionary of parameters for the
                current experimental run, such as `param_space_size`.
        """
        self.global_params = global_parameters

        self.verbose = self.global_params.verbose

        self.error_raise = self.global_params.error_raise

        self.ml_grid_object = ml_grid_object

        self.sub_sample_param_space_pct = self.global_params.sub_sample_param_space_pct

        self.parameter_space_size = local_param_dict.get("param_space_size")

        self.model_class_list = ml_grid_object.model_class_list

        if self.verbose >= 2:
            print(f"{len(self.model_class_list)} models loaded")

        self.pg_list = []

        for elem in self.model_class_list:
            
            if not self.global_params.bayessearch:
                pg = ParameterGrid(elem.parameter_space)
                pg = len(pg)
            else:
                # Handle list of parameter spaces , example log reg

                pg = calculate_combinations(elem.parameter_space, steps=10)
                

            #pg = ParameterGrid(elem.parameter_space)

            self.pg_list.append(pg)

            if self.verbose >= 1:
                print(f"{elem.method_name}:{pg}")

            for param in elem.parameter_space:
                
                if self.global_params.bayessearch is False:
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
                #validate bayes params?
                        
                        

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

    def execute(self) -> Tuple[List[List[Any]], float]:
        """Executes the grid search for each model in the list.

        This method iterates through the list of configured models and their
        parameter spaces, running a cross-validated grid search for each one.
        It captures any errors that occur during the process and returns a list
        of those errors along with the highest score achieved.

        Returns:
            Tuple[List[List[Any]], float]: A tuple containing:
                - A list of model errors, where each error is a list containing
                  the algorithm instance, the exception, and the traceback.
                - The highest score achieved across all successful model runs.
        """

        self.model_error_list = []
        
        highest_score = 0 # for optimisation

        if self.multiprocess:

            def multi_run_wrapper(args: Tuple) -> Any:
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
                    res = grid_search_cross_validate.grid_search_crossvalidate(
                        *self.arg_list[k]
                        # algorithm_implementation = LogisticRegression_class(parameter_space_size=self.parameter_space_size).algorithm_implementation, parameter_space = self.arg_list[k][1], method_name=self.arg_list[k][2], X = self.arg_list[k][3], y=self.arg_list[k][4]
                    ).grid_search_cross_validate_score_result
                    
                    highest_score = max(highest_score, res)
                    print(f"highest score: {highest_score}")

                except CatBoostError as e:
                    print(f"CatBoostError: {e}")
                    print(f"continuing despite catboost error...")

                except Exception as e:

                    print(e)
                    print("error on ", self.arg_list[k][2])
                    self.model_error_list.append(
                        [self.arg_list[k][0], e, traceback.print_exc()]
                    )

                    if self.error_raise:
                        raise e
                        res = input(
                            "error thrown in grid_search_crossvalidate on model class list, input pass to pass else raise"
                        )
                        if res == "pass":
                            continue
                        else:
                            raise e

        print(
            f"Model error list: nb. errors returned from func: {self.model_error_list}"
        )
        print(self.model_error_list)
        
        # return highest score from run for additional optimisation:
        

        return self.model_error_list, highest_score
