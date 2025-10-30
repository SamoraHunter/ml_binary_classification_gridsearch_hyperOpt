import unittest
import pkgutil
import importlib
import inspect
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from skopt import Optimizer  # Use the main Optimizer for sampling
from skopt.space import Real, Integer, Categorical  # Import dimension types
import ml_grid.model_classes
from ml_grid.util.global_params import global_parameters


class TestAllClassifierParamSpaces(unittest.TestCase):
    """
    Dynamically discovers and tests the parameter spaces for all classifier classes
    in the ml_grid.model_classes package.
    """

    def discover_classifier_classes(self):
        """
        Discovers all classifier modules and their main class.
        Yields the module name and the class object.
        """
        package = ml_grid.model_classes
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            if (
                module_name.startswith("test_")
                or module_name == "test_model_classes_param_spaces"
            ):
                continue

            full_module_name = f"{package.__name__}.{module_name}"
            module = importlib.import_module(full_module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Convention: Find the class defined within the module, not imported.
                if obj.__module__ == full_module_name:
                    # Exclude abstract base classes from direct testing
                    if "Base" in name:
                        continue

                    yield module_name, obj
                    break  # Assume one main class per file

    def test_all_classifier_param_spaces(self):
        """
        Main test method that iterates through all discovered classifiers
        and validates their grid and bayesian parameter spaces.
        """
        found_valid_classifier = False

        for module_name, classifier_class_def in self.discover_classifier_classes():
            with self.subTest(classifier=module_name):
                # Instantiate the class to check its structure.
                class_instance = self._instantiate_classifier(
                    classifier_class_def, parameter_space_size="small"
                )

                # If it's not a standard scikit-learn estimator, log and continue to the next.
                if not hasattr(class_instance, "algorithm_implementation"):
                    print(f"Skipping non-standard classifier: {module_name}")
                    continue

                found_valid_classifier = True
                # Validate both grid and bayesian search spaces
                self._validate_parameter_space(
                    classifier_class_def, module_name, is_bayes=False
                )
                self._validate_parameter_space(
                    classifier_class_def, module_name, is_bayes=True
                )

        self.assertTrue(
            found_valid_classifier,
            "No valid scikit-learn compatible classifiers were found and tested.",
        )

    def _instantiate_classifier(self, classifier_class_def, **kwargs):
        """
        Instantiates a classifier, passing only the arguments it accepts.
        This prevents TypeErrors for classifiers that don't take certain arguments.
        """
        sig = inspect.signature(classifier_class_def.__init__)

        # Special handling for classifiers requiring X and y at __init__
        if "X" in sig.parameters and "y" in sig.parameters:
            # Create dummy data for instantiation
            dummy_X = pd.DataFrame(np.random.rand(10, 2), columns=["a", "b"])
            dummy_y = pd.Series(np.random.randint(0, 2, 10))
            kwargs["X"] = dummy_X
            kwargs["y"] = dummy_y

        if "TabTransformer" in classifier_class_def.__name__:
            kwargs["categories"] = (2, 3)
            kwargs["num_continuous"] = 2

        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        class_instance = classifier_class_def(**valid_kwargs)

        # For wrappers around estimators that take an 'estimator' parameter (like AdaBoost),
        # set it on the underlying algorithm implementation directly.
        if "estimator" in kwargs and hasattr(
            class_instance.algorithm_implementation, "estimator"
        ):
            # This ensures the base estimator is correctly configured before validation.
            setattr(
                class_instance.algorithm_implementation,
                "estimator",
                kwargs["estimator"],
            )

        return class_instance

    def _validate_parameter_space(self, classifier_class_def, module_name, is_bayes):
        """
        A generic helper to validate a parameter space for a given classifier.

        Args:
            classifier_class_def: The classifier class definition.
            module_name: The name of the module for sub-testing.
            is_bayes: Boolean flag to determine which parameter space to test.
        """
        search_type = "bayesian" if is_bayes else "grid"
        with self.subTest(search_type=search_type, classifier=module_name):
            global_parameters.bayessearch = is_bayes
            # Instantiate the class once to get the correct parameter space
            class_instance = self._instantiate_classifier(
                classifier_class_def, parameter_space_size="small"
            )
            raw_param_space = (
                class_instance.parameter_space
            )  # Renamed to avoid confusion

            if is_bayes:
                # Normalize the parameter space to a list of dictionaries, similar to grid search.
                param_space_list = []
                if isinstance(raw_param_space, dict):
                    param_space_list = [raw_param_space]
                elif isinstance(raw_param_space, list):
                    param_space_list = raw_param_space

                # Iterate over each parameter grid in the list
                for grid_index, grid_dict in enumerate(param_space_list):
                    with self.subTest(grid_index=grid_index):
                        # Convert the dictionary-style grid_dict into skopt.space.Dimension objects
                        skopt_dimensions = []
                        for param_name, values in grid_dict.items():
                            if isinstance(values, np.ndarray):
                                if values.size == 0:
                                    continue
                            elif not values:
                                continue
                            if isinstance(values, (Real, Integer, Categorical)):
                                # If 'values' is already a skopt Dimension object, use it directly.
                                # Ensure it has a name, using param_name from the dict key if not already set.
                                if values.name is None:
                                    values.name = param_name
                                skopt_dimensions.append(values)
                            else:
                                # Otherwise, interpret as a list of categorical choices
                                # Convert numpy arrays and lists to tuples to make them hashable for skopt
                                processed_values = [
                                    tuple(v) if isinstance(v, (np.ndarray, list)) else v
                                    for v in values
                                ]
                                skopt_dimensions.append(
                                    Categorical(processed_values, name=param_name)
                                )

                        if not skopt_dimensions:
                            continue

                        # Sample N random parameter combinations for Bayesian search
                        num_samples = (
                            5  # Reduced for efficiency, as we might be in a loop
                        )
                        for i in range(num_samples):
                            optimizer = Optimizer(
                                dimensions=skopt_dimensions,
                                random_state=i,
                                base_estimator="dummy",
                            )
                            sampled_point = optimizer.ask()
                            params = dict(
                                zip(
                                    [dim.name for dim in skopt_dimensions],
                                    sampled_point,
                                )
                            )

                            init_kwargs = {"parameter_space_size": "small"}
                            if "estimator" in params:
                                init_kwargs["estimator"] = params["estimator"]

                            class_instance = self._instantiate_classifier(
                                classifier_class_def, **init_kwargs
                            )
                            base_estimator = class_instance.algorithm_implementation

                            with self.subTest(sample_num=i, params=params):
                                self._apply_and_validate_params(base_estimator, params)
            else:
                # Iterate through all combinations for Grid search
                # The parameter space for grid search can be a single dict (most common)
                # or a list of dicts (for complex cases like AdaBoost).
                # We normalize it to a list to handle both cases uniformly.
                param_space = None  # Initialize to ensure it's always defined
                if isinstance(raw_param_space, dict):
                    param_space = [raw_param_space]
                elif isinstance(raw_param_space, list):
                    param_space = raw_param_space

                self.assertIsInstance(
                    param_space,
                    list,
                    f"Grid search space for {module_name} should be a list of dicts",
                )

                for grid in param_space:
                    # For grids that specify the estimator, instantiate the class with it.
                    # This is crucial for models like AdaBoost where the `algorithm` parameter's
                    # validity depends on the `estimator`.
                    initial_estimator = None
                    if "estimator" in grid and len(grid["estimator"]) == 1:
                        initial_estimator = grid["estimator"][0]

                    # Only pass the estimator argument if it's relevant for the current grid.
                    if initial_estimator:
                        class_instance = self._instantiate_classifier(
                            classifier_class_def,
                            estimator=initial_estimator,
                            parameter_space_size="small",
                        )
                    else:
                        class_instance = self._instantiate_classifier(
                            classifier_class_def, parameter_space_size="small"
                        )
                    base_estimator = class_instance.algorithm_implementation

                    # Create a smaller, targeted grid for testing.
                    # For numeric parameters, we test the min and max values.
                    # For string/categorical parameters, we test all combinations.
                    reduced_grid = {}
                    for param, values in grid.items():
                        if isinstance(values, np.ndarray):
                            if values.size == 0:
                                continue
                        elif not values:
                            continue
                        # Check if all values are numeric (int or float)
                        is_numeric = all(isinstance(v, (int, float)) for v in values)
                        if is_numeric and len(values) > 2:
                            reduced_grid[param] = [min(values), max(values)]
                        else:
                            # Keep all values for categorical or short lists
                            reduced_grid[param] = values

                    if not reduced_grid:
                        continue

                    # Now, iterate through all combinations of the *reduced* grid.
                    # This is more thorough than random sampling but more efficient
                    # than testing the full original grid.
                    param_grid_list = list(ParameterGrid(reduced_grid))

                    for i, params in enumerate(param_grid_list):
                        with self.subTest(grid_combination_num=i, params=params):
                            # We might need to re-instantiate if the estimator is part of the grid
                            if "estimator" in params:
                                class_instance = self._instantiate_classifier(
                                    classifier_class_def,
                                    estimator=params["estimator"],
                                    parameter_space_size="small",
                                )
                                base_estimator = class_instance.algorithm_implementation
                            self._apply_and_validate_params(base_estimator, params)

    def _apply_and_validate_params(self, base_estimator, params):
        """Clones an estimator, applies parameters, and validates them."""
        estimator_clone = clone(base_estimator)

        # Create a mutable copy of params to modify
        current_params = params.copy()

        # The estimator object itself is not a valid parameter for set_params,
        # as it's handled during initialization. Remove it before setting other params.
        if "estimator" in current_params:
            del current_params["estimator"]

        if current_params:
            estimator_clone.set_params(**current_params)

        # Some custom or older sklearn-compatible estimators might not have _parameter_constraints.
        # In that case, we can't use the built-in validation, but we can still confirm
        # that the parameters were set without error.
        if hasattr(estimator_clone, "_validate_params") and hasattr(
            estimator_clone, "_parameter_constraints"
        ):
            # This will raise InvalidParameterError on failure for modern estimators.
            estimator_clone._validate_params()


if __name__ == "__main__":
    unittest.main()
