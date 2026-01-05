import logging
import traceback
import glob
import os
import yaml
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import ParameterGrid

from ml_grid.pipeline import grid_search_cross_validate
from ml_grid.pipeline.data import pipe
from ml_grid.util.bayes_utils import calculate_combinations
from ml_grid.util.global_params import global_parameters
from ml_grid.util.project_score_save import project_score_save_class  # Import the class


class run:
    """Orchestrates the hyperparameter search for a list of models."""

    global_params: global_parameters
    """A reference to the global parameters singleton instance."""

    verbose: int
    """The verbosity level for logging, inherited from global parameters."""

    error_raise: bool
    """A flag to control error handling. If True, exceptions will be raised."""

    ml_grid_object: pipe
    """The main data pipeline object, containing data and model configurations."""

    sub_sample_param_space_pct: float
    """The percentage of the parameter space to sample in a randomized search."""

    parameter_space_size: str
    """The size of the parameter space for base learners (e.g., 'medium', 'xsmall')."""

    model_class_list: List[Any]
    """A list of instantiated model class objects to be evaluated in this run."""

    pg_list: List[int]
    """A list containing the calculated size of the parameter grid for each model."""

    mean_parameter_space_val: float
    """The mean size of the parameter spaces across all models in the run."""

    sub_sample_parameter_val: int
    """The calculated number of iterations for randomized search, based on `sub_sample_param_space_pct`."""

    arg_list: List[Tuple]
    """A list of argument tuples, one for each model, to be passed to the grid search function."""

    multiprocess: bool
    """A flag to enable or disable multiprocessing for running grid searches in parallel."""

    local_param_dict: Dict[str, Any]
    """A dictionary of parameters for the current experimental run."""

    model_error_list: List[List[Any]]
    """A list to store details of any errors encountered during model training."""

    highest_score: float
    """The highest score achieved across all successful model runs in the execute step."""

    def __init__(self, local_param_dict: Dict[str, Any], **kwargs):
        """Initializes the run class.

        This class takes the main data pipeline object and a dictionary of local
        parameters to set up and prepare for executing a series of hyperparameter
        searches across multiple machine learning models.

        For hyperopt, this constructor can also accept keyword arguments to
        create the `pipe` object internally.

        Args:
            local_param_dict (Dict[str, Any]): A dictionary of parameters for the
                current experimental run, such as `param_space_size`.
            **kwargs: Keyword arguments to be passed to the `pipe` constructor.
                Expected keys include `file_name`, `drop_term_list`, `model_class_dict`,
                `base_project_dir`, `experiment_dir`, and `outcome_var`.
        """
        self.global_params = global_parameters

        # Update global parameters if provided in kwargs
        if "global_params" in kwargs and isinstance(kwargs["global_params"], dict):
            self.global_params.update_parameters(**kwargs["global_params"])

        self.logger = logging.getLogger("ml_grid")

        self.verbose = self.global_params.verbose

        if "ml_grid_object" in kwargs:
            self.ml_grid_object = kwargs["ml_grid_object"]
        else:
            # Create the pipe object from the provided kwargs
            pipe_kwargs = {
                "file_name": kwargs.get("file_name"),
                "drop_term_list": kwargs.get("drop_term_list"),
                "model_class_dict": kwargs.get("model_class_dict"),
                "local_param_dict": local_param_dict,
                "base_project_dir": kwargs.get("base_project_dir"),
                "experiment_dir": kwargs.get("experiment_dir"),
                "outcome_var": kwargs.get("outcome_var"),
                "param_space_index": kwargs.get("param_space_index", 0),
            }
            self.ml_grid_object = pipe(**pipe_kwargs)

        # Propagate n_iter from global_params to local_param_dict
        # This ensures the value persists across process boundaries (pickling) where the singleton might be reset
        self.ml_grid_object.local_param_dict["n_iter"] = self.global_params.n_iter

        self.error_raise = self.global_params.error_raise

        self.sub_sample_param_space_pct = self.global_params.sub_sample_param_space_pct

        self.parameter_space_size = local_param_dict.get("param_space_size")

        self.model_class_list = self.ml_grid_object.model_class_list

        if self.verbose >= 2:
            self.logger.info(f"{len(self.model_class_list)} models loaded")

        self.pg_list = []

        for elem in self.model_class_list:

            if not self.global_params.bayessearch:
                # ParameterGrid can now be called directly, as the model class
                # provides a grid-search-compatible parameter space.
                pg = ParameterGrid(elem.parameter_space)
                pg = len(pg)
            else:

                pg = calculate_combinations(elem.parameter_space, steps=10)

            # pg = ParameterGrid(elem.parameter_space)

            self.pg_list.append(pg)

            if self.verbose >= 1:
                self.logger.info(f"{elem.method_name} parameter space size: {pg}")

            # Determine if parameter_space is a list of dicts or a single dict
            param_dicts = (
                elem.parameter_space
                if isinstance(elem.parameter_space, list)
                else [elem.parameter_space]
            )

            for param_dict in param_dicts:
                if not isinstance(param_dict, dict):
                    continue

                for param_key in param_dict:
                    if self.global_params.bayessearch is False:
                        try:
                            param_value = param_dict.get(param_key)
                            if not isinstance(param_value, list) and not isinstance(
                                param_value, np.ndarray
                            ):
                                self.logger.warning(
                                    "Unexpected parameter type in grid search space."
                                )
                                self.logger.warning(
                                    f"{elem.method_name, param_key} {type(param_value)}"
                                )

                        except (AttributeError, TypeError, KeyError) as e:
                            self.logger.error(
                                f"Error validating parameters for {elem.method_name}: {e}",
                                exc_info=True,
                            )
                            if self.error_raise:
                                self.logger.critical(
                                    "Halting execution due to parameter validation error as 'error_raise' is True."
                                )
                                raise
                            else:
                                self.logger.warning(
                                    "Continuing despite parameter validation error as 'error_raise' is False."
                                )
                    # validate bayes params?

        # sample from mean of all param space n
        if self.pg_list:
            self.mean_parameter_space_val = np.mean(self.pg_list)
            self.sub_sample_parameter_val = int(
                self.sub_sample_param_space_pct * self.mean_parameter_space_val
            )
        else:
            self.logger.warning(
                "Parameter grid list is empty; no models were loaded. Setting parameter space values to 0."
            )
            self.mean_parameter_space_val = 0
            self.sub_sample_parameter_val = 0

        # Initialize the project_score_save_class instance once per run
        # The ml_grid_object should have the experiment_dir set
        self.project_score_save_class_instance = project_score_save_class(
            experiment_dir=self.ml_grid_object.experiment_dir
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
                    self.project_score_save_class_instance,  # Pass the instance here
                )
            )

        self.multiprocess = False

        self.local_param_dict = local_param_dict

        if self.verbose >= 2:
            self.logger.info(f"Passed main init, len(arg_list): {len(self.arg_list)}")

    def _prepare_run(self, model_class):
        """Prepares a single model run by creating the necessary arguments."""
        return (
            model_class.algorithm_implementation,
            model_class.parameter_space,
            model_class.method_name,
            self.ml_grid_object,
            self.sub_sample_parameter_val,
            self.project_score_save_class_instance,
        )

    def execute_single_model(self, args: Tuple) -> float:
        """
        Executes the grid search for a single model and returns its score.
        This method is designed to be called within a hyperopt objective function.
        """
        try:
            self.logger.info(f"Starting grid search for {args[2]}...")
            gscv_instance = grid_search_cross_validate.grid_search_crossvalidate(*args)
            score = gscv_instance.grid_search_cross_validate_score_result
            self.logger.info(f"Score for {args[2]}: {score:.4f}")
            return score

        except Exception as e:
            self.logger.error(
                f"An exception occurred during grid search for {args[2]}: {e}",
                exc_info=True,
            )
            self.model_error_list.append([args[0], e, traceback.format_exc()])
            if self.error_raise:
                self.logger.critical("Halting due to 'error_raise' flag.")
                raise
            else:
                self.logger.warning("Continuing as 'error_raise' is False.")
                return 0.0  # Return a poor score on failure

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
        self.highest_score = 0
        highest_score = 0  # for optimisation

        if self.multiprocess:

            def multi_run_wrapper(args: Tuple) -> Any:
                self.logger.warning("Multiprocessing is not fully implemented.")
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
                    self.logger.info(
                        f"Starting grid search for {self.arg_list[k][2]}..."
                    )
                    gscv_instance = (
                        grid_search_cross_validate.grid_search_crossvalidate(
                            *self.arg_list[k]  # Unpack all arguments
                        )
                    )

                    self.highest_score = max(
                        self.highest_score,
                        gscv_instance.grid_search_cross_validate_score_result,
                    )
                    self.logger.info(f"Current highest score: {self.highest_score:.4f}")

                except (
                    Exception
                ) as e:  # Catches any exception from grid_search_crossvalidate
                    self.logger.error(
                        f"An exception occurred during grid search for {self.arg_list[k][2]}: {e}",
                        exc_info=True,
                    )

                    self.model_error_list.append(
                        [self.arg_list[k][0], e, traceback.format_exc()]
                    )

                    # Based on the 'error_raise' flag, either halt execution or log and continue.
                    if self.error_raise:
                        self.logger.critical(
                            "Halting execution due to an exception during model run as 'error_raise' is True."
                        )
                        raise
                    else:
                        self.logger.warning(
                            f"Caught exception for {self.arg_list[k][2]} and continuing as 'error_raise' is False."
                        )

        self.logger.info(
            f"Model error list: nb. errors returned from func: {self.model_error_list}"
        )

        # return highest score from run for additional optimisation:

        return self.model_error_list, self.highest_score
