import traceback
import numpy as np
from ml_grid.pipeline import grid_search_cross_validate
from ml_grid.util import grid_param_space
from ml_grid.util.global_params import global_parameters
from sklearn.model_selection import ParameterGrid
from ml_grid.util.bayes_utils import calculate_combinations


class run:

    def __init__(self, ml_grid_object, local_param_dict):  # kwargs**

        self.global_params = global_parameters()

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

    def execute(self):

        # needs implementing*

        self.model_error_list = []
        
        highest_score = 0 # for optimisation

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
                    res = grid_search_cross_validate.grid_search_crossvalidate(
                        *self.arg_list[k]
                        # algorithm_implementation = LogisticRegression_class(parameter_space_size=self.parameter_space_size).algorithm_implementation, parameter_space = self.arg_list[k][1], method_name=self.arg_list[k][2], X = self.arg_list[k][3], y=self.arg_list[k][4]
                    ).grid_search_cross_validate_score_result
                    
                    highest_score = max(highest_score, res[0])
                    print(f"highest score: {highest_score}")

                except Exception as e:

                    print(e)
                    print("error on ", self.arg_list[k][2])
                    self.model_error_list.append(
                        [self.arg_list[k][0], e, traceback.print_exc()]
                    )

                    if self.error_raise:
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
