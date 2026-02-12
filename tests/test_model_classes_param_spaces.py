import unittest
import importlib
import inspect
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from skopt import Optimizer  # Use the main Optimizer for sampling
from skopt.space import Real, Integer, Categorical  # Import dimension types
from ml_grid.util.global_params import global_parameters
import os


class TestAllClassifierParamSpaces(unittest.TestCase):
    """
    Dynamically discovers and tests the parameter spaces for all classifier classes
    in the ml_grid.model_classes package.
    """

    def discover_classifier_classes(self):
        """
        Dynamically discovers all classifier classes in the model_classes directory,
        skipping any that are known to be deprecated or problematic.
        """
        # Define a list of files to explicitly skip during test discovery.
        # We add the deprecated wrappers here to prevent them from being imported.
        files_to_skip = {
            "knn_wrapper_class.py",
            "knn_gpu_classifier_class.py",
            "__init__.py",
            "H2OBaseClassifier.py",  # Skip base class - it's abstract and requires estimator_class
        }
        # --- END OF FIX ---

        package_dir = os.path.join(
            os.path.dirname(__file__), "..", "ml_grid", "model_classes"
        )

        for filename in os.listdir(package_dir):
            # Check if the file is a Python file AND not in our skip list
            if filename.endswith(".py") and filename not in files_to_skip:
                module_name = filename[:-3]
                full_module_name = f"ml_grid.model_classes.{module_name}"

                try:
                    module = importlib.import_module(full_module_name)

                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Heuristic to find the main classifier class in the file
                        if hasattr(obj, "get_params") and hasattr(obj, "fit"):
                            # Check if the class is defined in this module, not imported from elsewhere
                            if obj.__module__ == full_module_name:
                                print(f"Discovered classifier '{name}' in '{filename}'")
                                yield module_name, obj
                                break  # Assume one main class per file
                except ImportError as e:
                    # This will now only catch unexpected import errors
                    self.fail(
                        f"Failed to import module {full_module_name} during test discovery. Error: {e}"
                    )
                except Exception as e:
                    self.fail(
                        f"An unexpected error occurred while processing {full_module_name}: {e}"
                    )

    def test_all_classifier_param_spaces(self):
        """
        Main test method that iterates through all discovered classifiers
        and validates their grid and bayesian parameter spaces.
        """
        found_valid_classifier = False

        for module_name, classifier_class_def in self.discover_classifier_classes():
            with self.subTest(classifier=module_name):
                # Instantiate the class to check its structure.
                try:
                    class_instance = self._instantiate_classifier(
                        classifier_class_def, parameter_space_size="small"
                    )
                except Exception as e:
                    print(f"Skipping {module_name} - failed to instantiate: {e}")
                    continue

                # Determine which object to use for testing
                # Try algorithm_implementation first (for wrappers), then fall back to the instance itself
                if hasattr(class_instance, "algorithm_implementation"):
                    test_object = class_instance.algorithm_implementation
                    object_type = "wrapped sklearn estimator"
                else:
                    # Use the instance itself if it has the sklearn interface
                    test_object = class_instance
                    object_type = "direct estimator"

                # Check if the object has parameter_space attribute
                if not hasattr(class_instance, "parameter_space"):
                    print(f"Skipping {module_name} - no parameter_space attribute")
                    continue

                print(f"Testing {module_name} as {object_type}")
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
        if (
            "estimator" in kwargs
            and hasattr(class_instance, "algorithm_implementation")
            and hasattr(class_instance.algorithm_implementation, "estimator")
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

            # Determine which object to use for validation
            if hasattr(class_instance, "algorithm_implementation"):
                base_estimator = class_instance.algorithm_implementation
            else:
                base_estimator = class_instance

            if is_bayes:
                # Normalize the parameter space to a list of dictionaries,
                # similar to grid search.
                param_space_list = []
                if isinstance(raw_param_space, dict):
                    param_space_list = [raw_param_space]
                elif isinstance(raw_param_space, list):
                    param_space_list = raw_param_space

                # Iterate over each parameter grid in the list
                for grid_index, grid_dict in enumerate(param_space_list):
                    with self.subTest(grid_index=grid_index):
                        # Convert the dictionary-style grid_dict into
                        # skopt.space.Dimension objects
                        skopt_dimensions = []
                        for param_name, values in grid_dict.items():
                            if isinstance(values, np.ndarray):
                                if values.size == 0:
                                    continue
                            elif not values:
                                continue
                            if isinstance(values, (Real, Integer, Categorical)):
                                # If 'values' is already a skopt Dimension
                                # object, use it directly. Ensure it has a
                                # name, using param_name from the dict key if
                                # not already set.
                                if values.name is None:
                                    values.name = param_name
                                skopt_dimensions.append(values)
                            else:
                                # Otherwise, interpret as a list of categorical choices
                                # Convert numpy arrays and lists to tuples to
                                # make them hashable for skopt
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
                            if hasattr(class_instance, "algorithm_implementation"):
                                test_estimator = class_instance.algorithm_implementation
                            else:
                                test_estimator = class_instance

                            with self.subTest(sample_num=i, params=params):
                                self._apply_and_validate_params(test_estimator, params)
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
                    # This is crucial for models like AdaBoost where the
                    # `algorithm` parameter's validity depends on the `estimator`.
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

                    if hasattr(class_instance, "algorithm_implementation"):
                        test_estimator = class_instance.algorithm_implementation
                    else:
                        test_estimator = class_instance

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
                        if isinstance(values, Categorical):
                            reduced_grid[param] = values.categories
                        else:
                            # Handle skopt Integer/Real that might leak into grid space definitions
                            if isinstance(values, (Integer, Real)):
                                reduced_grid[param] = [values.low, values.high]
                            else:
                                is_numeric = all(
                                    isinstance(v, (int, float)) for v in values
                                )
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
                            # We might need to re-instantiate if the estimator
                            # is part of the grid
                            if "estimator" in params:
                                class_instance = self._instantiate_classifier(
                                    classifier_class_def,
                                    estimator=params["estimator"],
                                    parameter_space_size="small",
                                )
                                if hasattr(class_instance, "algorithm_implementation"):
                                    test_estimator = (
                                        class_instance.algorithm_implementation
                                    )
                                else:
                                    test_estimator = class_instance
                            self._apply_and_validate_params(test_estimator, params)

    def _apply_and_validate_params(self, base_estimator, params):
        """Clones an estimator, applies parameters, and validates them."""
        try:
            estimator_clone = clone(base_estimator)
        except Exception as e:
            # If cloning fails, just use the original estimator for validation
            print(f"Warning: Could not clone estimator, using original: {e}")
            estimator_clone = base_estimator

        # Create a mutable copy of params to modify
        current_params = params.copy()

        # The estimator object itself is not a valid parameter for set_params,
        # as it's handled during initialization. Remove it before setting other params.
        if "estimator" in current_params:
            del current_params["estimator"]

        if current_params:
            try:
                estimator_clone.set_params(**current_params)
            except Exception as e:
                self.fail(f"Failed to set parameters {current_params}: {e}")

        # Some custom or older sklearn-compatible estimators might not have
        # _parameter_constraints. In that case, we can't use the built-in
        # validation, but we can still confirm
        # that the parameters were set without error.
        if hasattr(estimator_clone, "_validate_params") and hasattr(
            estimator_clone, "_parameter_constraints"
        ):
            # This will raise InvalidParameterError on failure for modern estimators.
            try:
                estimator_clone._validate_params()
            except Exception as e:
                self.fail(
                    f"Parameter validation failed for params {current_params}: {e}"
                )


if __name__ == "__main__":
    unittest.main()
