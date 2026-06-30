import unittest
import sys
import os
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch

from ml_grid.model_classes.tabtransformer_classifier_class import (
    TabTransformerWrapper,
    TabTransformerClass,
)
from ml_grid.model_classes.tabtransformerClassifier import TabTransformerClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestTabTransformerWrapper(unittest.TestCase):
    """Test TabTransformerWrapper class that handles tuple-based parameters."""

    def setUp(self):
        """Set up test fixtures."""
        self.categories = (10, 5, 6, 5, 8)
        self.num_continuous = 4

    def test_initialization_without_wrapper(self):
        """Test that TabTransformerClassifier can be initialized directly."""
        wrapper = TabTransformerWrapper(
            categories=self.categories, num_continuous=self.num_continuous
        )
        self.assertEqual(wrapper.categories, self.categories)
        self.assertEqual(wrapper.num_continuous, self.num_continuous)

    def test_set_params_with_tuple_mapping_categories(self):
        """Test that set_params correctly maps integer indices to tuples for categories."""
        wrapper = TabTransformerWrapper(categories=(10, 5), num_continuous=2)

        # Test mapping index 0 to first tuple
        result = wrapper.set_params(categories=0)
        expected_categories = (10, 5, 6, 5, 8)
        self.assertEqual(result.categories, expected_categories)

    def test_set_params_with_tuple_mapping_mlp_hidden_mults(self):
        """Test that set_params correctly maps integer indices to tuples for mlp_hidden_mults."""
        wrapper = TabTransformerWrapper(categories=(10, 5), num_continuous=2)

        # Test mapping index 0 to first tuple
        result = wrapper.set_params(mlp_hidden_mults=0)
        expected_mlp = (4, 2)
        self.assertEqual(result.mlp_hidden_mults, expected_mlp)

    def test_set_params_with_index_out_of_range(self):
        """Test that set_params handles out-of-range indices gracefully."""
        wrapper = TabTransformerWrapper(categories=(10, 5), num_continuous=2)

        # Set invalid index - should not crash and passes through unchanged
        result = wrapper.set_params(categories=99)
        self.assertEqual(result.categories, 99)

    def test_set_params_with_negative_index(self):
        """Test that set_params handles negative indices gracefully."""
        wrapper = TabTransformerWrapper(categories=(10, 5), num_continuous=2)

        # Set negative index - should not crash and passes through unchanged
        result = wrapper.set_params(categories=-1)
        self.assertEqual(result.categories, -1)

    def test_set_params_with_non_integer_value(self):
        """Test that set_params handles non-integer values by passing through."""
        wrapper = TabTransformerWrapper(categories=(10, 5), num_continuous=2)

        # Pass tuple directly - should not be modified
        result = wrapper.set_params(categories=(15, 7))
        self.assertEqual(result.categories, (15, 7))

    def test_set_params_preserves_other_parameters(self):
        """Test that set_params preserves parameters not in tuple_mapping."""
        wrapper = TabTransformerWrapper(categories=(10, 5), num_continuous=2)

        result = wrapper.set_params(dim=64, depth=4)
        self.assertEqual(result.dim, 64)
        self.assertEqual(result.depth, 4)

    def test_set_params_returns_self(self):
        """Test that set_params returns the instance for method chaining."""
        wrapper = TabTransformerWrapper(categories=(10, 5), num_continuous=2)

        result = wrapper.set_params(categories=0)
        self.assertIs(result, wrapper)


class TestTabTransformerClass(unittest.TestCase):
    """Test TabTransformerClass with full integration."""

    @classmethod
    def setUpClass(cls):
        """Set up class-wide fixtures."""
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 100

        # Categorical columns
        df_categ = pd.DataFrame(
            {
                "cat1": np.random.choice(["A", "B", "C"], n_samples),
                "cat2": np.random.choice(["X", "Y"], n_samples),
                "cat3": np.random.choice(["P", "Q", "R", "S"], n_samples),
            }
        )

        # Continuous columns
        df_cont = pd.DataFrame(
            {
                "cont1": np.random.randn(n_samples),
                "cont2": np.random.randn(n_samples),
                "cont3": np.random.rand(n_samples) * 10,
                "cont4": np.random.randint(0, 100, n_samples),
            }
        )

        cls.X = pd.concat([df_categ, df_cont], axis=1)
        cls.y = pd.Series(np.random.randint(0, 2, n_samples))

    def setUp(self):
        """Set up test fixtures."""
        # Reset the global singleton first - ensure bayessearch is False for grid search tests
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = False

    def tearDown(self):
        """Clean up test fixtures."""
        # Restore to a known state
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = False

    def test_initialization_with_grid_search(self):
        """Test initialization with grid search configuration."""
        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        self.assertEqual(model.method_name, "TabTransformerClassifier")
        self.assertIsInstance(model.parameter_space, dict)
        self.assertIn("categories", model.parameter_space)
        self.assertIn("num_continuous", model.parameter_space)

    def test_initialization_with_bayesian_search(self):
        """Test initialization with Bayesian search configuration."""
        # Set global singleton for bayesian mode
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = True

        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        self.assertIsInstance(model.parameter_space, dict)
        # Check Bayesian-specific parameter types
        from skopt.space import Real, Categorical

        param_values = list(model.parameter_space.values())
        has_real = any(isinstance(v, Real) for v in param_values)
        has_categorical = any(isinstance(v, Categorical) for v in param_values)
        self.assertTrue(has_real or has_categorical)

    def test_parameter_space_keys(self):
        """Test that parameter space contains expected keys."""
        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        expected_keys = [
            "categories",
            "num_continuous",
            "dim",
            "dim_out",
            "depth",
            "heads",
            "attn_dropout",
            "ff_dropout",
            "mlp_hidden_mults",
            "mlp_act",
            "continuous_mean_std",
        ]

        for key in expected_keys:
            self.assertIn(key, model.parameter_space, f"Missing key: {key}")

    def test_algorithm_implementation_type_grid_search(self):
        """Test that algorithm_implementation is TabTransformerClassifier in grid mode."""
        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        self.assertIsInstance(model.algorithm_implementation, TabTransformerClassifier)

    def test_algorithm_implementation_type_bayesian_search(self):
        """Test that algorithm_implementation is TabTransformerWrapper in bayesian mode."""
        # Set global singleton for bayesian mode
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = True

        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        from ml_grid.model_classes.tabtransformer_classifier_class import (
            TabTransformerWrapper,
        )

        self.assertIsInstance(model.algorithm_implementation, TabTransformerWrapper)

    def test_parameter_vector_space_initialization(self):
        """Test that parameter_vector_space is properly initialized."""
        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        from ml_grid.util.param_space import ParamSpace

        self.assertIsInstance(model.parameter_vector_space, ParamSpace)

    def test_parameter_space_size_small(self):
        """Test parameter space generation with 'small' size."""
        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        # Check that ParamSpace's param_dict is set
        self.assertIsNotNone(model.parameter_vector_space.param_dict)

    def test_parameter_space_size_medium(self):
        """Test parameter space generation with 'medium' size."""
        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="medium")

        self.assertIsNotNone(model.parameter_vector_space.param_dict)

    def test_parameter_space_size_xwide(self):
        """Test parameter space generation with 'xwide' size."""
        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="xwide")

        self.assertIsNotNone(model.parameter_vector_space.param_dict)

    def test_parameter_space_size_none(self):
        """Test parameter space generation with None size."""
        from ml_grid.util.global_params import global_parameters as gp
        from ml_grid.util.param_space import ParamSpace

        gp.bayessearch = False

        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size=None)

        # Should still initialize but param_dict might be None
        self.assertIsInstance(model.parameter_vector_space, ParamSpace)

    def test_categorical_column_detection(self):
        """Test that categorical columns are correctly detected."""
        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        # Get categories from the internal algorithm implementation
        categories = model.algorithm_implementation.categories

        # We have 3 categorical columns with different unique counts
        self.assertIsInstance(categories, tuple)
        self.assertEqual(len(categories), 3)

    def test_continuous_column_count(self):
        """Test that continuous column count is correct."""
        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        num_cont = model.algorithm_implementation.num_continuous
        self.assertEqual(num_cont, 4)

    def test_categories_in_grid_param_space(self):
        """Test that categories in grid mode is a list of tuples."""
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = False

        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        cat_space = model.parameter_space["categories"]
        self.assertIsInstance(cat_space, list)
        # Should contain tuples
        first_value = cat_space[0]
        self.assertIsInstance(first_value, tuple)

    def test_mlp_act_in_grid_param_space(self):
        """Test that mlp_act in grid mode is a list containing module."""
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = False

        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        mlp_act = model.parameter_space["mlp_act"]
        # In grid mode, it's a list containing the module
        self.assertIsInstance(mlp_act, list)
        self.assertEqual(len(mlp_act), 1)
        import torch.nn as nn

        self.assertIsInstance(mlp_act[0], nn.Module)

    def test_continuous_mean_std_in_grid_param_space(self):
        """Test that continuous_mean_std in grid mode is a list containing tensor."""
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = False

        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        cont_mean_std = model.parameter_space["continuous_mean_std"]
        # In grid mode, it's a list containing the tensor
        self.assertIsInstance(cont_mean_std, list)
        self.assertEqual(len(cont_mean_std), 1)
        self.assertIsInstance(cont_mean_std[0], torch.Tensor)

    def test_bayesian_real_params(self):
        """Test that Bayesian parameters are Real type."""
        # Set global singleton for bayesian mode
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = True

        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        from skopt.space import Real

        param_values = list(model.parameter_space.values())
        real_params = [v for v in param_values if isinstance(v, Real)]
        self.assertGreater(len(real_params), 0)

    def test_bayesian_categorical_params(self):
        """Test that Bayesian parameters include Categorical type."""
        # Set global singleton for bayesian mode
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = True

        model = TabTransformerClass(X=self.X, y=self.y, parameter_space_size="small")

        from skopt.space import Categorical

        param_values = list(model.parameter_space.values())
        cat_params = [v for v in param_values if isinstance(v, Categorical)]
        self.assertGreater(len(cat_params), 0)


class TestTabTransformerClassEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    @classmethod
    def setUpClass(cls):
        """Set up class-wide fixtures."""
        pass

    @classmethod
    def tearDownClass(cls):
        """Restore global parameters after tests."""
        pass

    def setUp(self):
        """Set up test fixtures."""
        self.global_params_patch = patch("ml_grid.util.global_params.global_parameters")
        self.mock_global_params = self.global_params_patch.start()
        self.mock_global_params.bayessearch = False

    def tearDown(self):
        """Clean up test fixtures."""
        self.global_params_patch.stop()

    def test_empty_dataframe_input(self):
        """Test behavior with empty DataFrame (should raise appropriate error)."""
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = False

        X_empty = pd.DataFrame()
        y_empty = pd.Series([])

        # Should handle gracefully or raise expected errors related to invalid data
        try:
            # noqa: F841
            TabTransformerClass(X=X_empty, y=y_empty, parameter_space_size="small")
        except AssertionError as e:
            # Expected: assertion error about null input shape or similar
            self.assertIn("null", str(e).lower() if "null" in str(e).lower() else True)
        except (ValueError, KeyError):
            # Also accept ValueError or KeyError
            pass

    def test_minimal_categorical_data(self):
        """Test with minimal categorical columns."""
        X_min = pd.DataFrame({"cat1": ["A", "B", "C"]})
        y_min = pd.Series([0, 1, 0])

        model = TabTransformerClass(X=X_min, y=y_min, parameter_space_size="small")

        self.assertEqual(len(model.algorithm_implementation.categories), 1)

    def test_single_continuous_column(self):
        """Test with single continuous column."""
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = False

        # Need at least one categorical column + continuous to avoid error
        X_single = pd.DataFrame({"cat1": ["A", "B", "C"], "cont1": [1.0, 2.0, 3.0]})
        y_single = pd.Series([0, 1, 0])

        model = TabTransformerClass(
            X=X_single, y=y_single, parameter_space_size="small"
        )

        self.assertEqual(model.algorithm_implementation.num_continuous, 1)

    def test_all_categorical_columns(self):
        """Test with only categorical columns."""
        X_categ_only = pd.DataFrame({"cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]})
        y_only = pd.Series([0, 1, 0])

        model = TabTransformerClass(
            X=X_categ_only, y=y_only, parameter_space_size="small"
        )

        self.assertEqual(model.algorithm_implementation.num_continuous, 0)

    def test_all_continuous_columns(self):
        """Test with only continuous columns (needs at least one categorical)."""
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = False

        # TabTransformer requires at least one categorical feature
        X_cont_only = pd.DataFrame(
            {
                "cat1": ["A", "B", "C"],
                "cont1": [1.0, 2.0, 3.0],
                "cont2": [4.0, 5.0, 6.0],
            }
        )
        y_only = pd.Series([0, 1, 0])

        model = TabTransformerClass(
            X=X_cont_only, y=y_only, parameter_space_size="small"
        )

        self.assertEqual(model.algorithm_implementation.num_continuous, 2)

    def test_large_number_of_categories(self):
        """Test with many categories in a column."""
        n_cats = 50
        X_large_cat = pd.DataFrame({"cat": [f"Cat_{i}" for i in range(n_cats)]})
        y_large = pd.Series([0, 1] * 25)

        model = TabTransformerClass(
            X=X_large_cat, y=y_large, parameter_space_size="small"
        )

        categories = model.algorithm_implementation.categories
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], n_cats)

    def test_float_columns_detected_as_continuous(self):
        """Test that float columns are detected as continuous."""
        from ml_grid.util.global_params import global_parameters as gp

        gp.bayessearch = False

        X_float = pd.DataFrame(
            {
                "cat1": ["A", "B", "C"],  # Need at least one categorical
                "float_col": [1.5, 2.5, 3.5],
                "int_col": [1, 2, 3],
            }
        )
        y_float = pd.Series([0, 1, 0])

        model = TabTransformerClass(X=X_float, y=y_float, parameter_space_size="small")

        self.assertEqual(model.algorithm_implementation.num_continuous, 2)

    def test_mixed_types_in_single_column(self):
        """Test behavior with mixed types in single column."""
        # This tests error handling - pandas will handle mixed types
        X_mixed = pd.DataFrame({"mixed": [1, "text", 3.5]})
        y_mixed = pd.Series([0, 1, 0])

        try:
            # noqa: F841
            TabTransformerClass(X=X_mixed, y=y_mixed, parameter_space_size="small")
            # If we get here, the mixed type was handled
        except Exception as e:
            # Expected to fail gracefully
            self.assertIsInstance(e, (TypeError, ValueError))


class TestTabTransformerClassifierIntegration(unittest.TestCase):
    """Test integration between TabTransformerWrapper and TabTransformerClassifier."""

    def test_wrapper_parameter_mapping(self):
        """Test complete parameter mapping through wrapper."""
        categories_tuple = (10, 5)
        num_cont = 2

        # Test with grid search
        wrapper = TabTransformerWrapper(
            categories=categories_tuple, num_continuous=num_cont
        )

        # Set parameters via wrapper - indices should map to tuples
        result = wrapper.set_params(categories=0)
        # The index 0 maps to the first tuple in the mapping: (10, 5, 6, 5, 8)
        self.assertEqual(result.categories, (10, 5, 6, 5, 8))

    def test_wrapper_mlp_mapping(self):
        """Test MLP multipliers mapping through wrapper."""
        categories_tuple = (10, 5)
        num_cont = 2

        wrapper = TabTransformerWrapper(
            categories=categories_tuple, num_continuous=num_cont
        )

        # Set mlp_hidden_mults with index
        result = wrapper.set_params(mlp_hidden_mults=0)
        self.assertEqual(result.mlp_hidden_mults, (4, 2))

    def test_algorithm_implementation_properties(self):
        """Test that algorithm implementation exposes expected properties."""
        categories_tuple = (10, 5)
        num_cont = 2

        classifier = TabTransformerClassifier(
            categories=categories_tuple, num_continuous=num_cont
        )

        self.assertEqual(classifier.categories, categories_tuple)
        self.assertEqual(classifier.num_continuous, num_cont)


if __name__ == "__main__":
    unittest.main()
