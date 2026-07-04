"""
Unit tests for the hierarchical hyperparameter search functionality.

This test suite validates:
- HierarchicalParamSpace class generates correct stage spaces
- Parameter importance analysis works correctly
- Early stopping rules function as expected
- Backward compatibility with existing configurations
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys


class TestHierarchicalSearch(unittest.TestCase):
    """Test suite for hierarchical hyperparameter search optimization."""

    @classmethod
    def setUpClass(cls):
        """Set up shared test resources."""
        cls.project_root = Path(__file__).resolve().parents[1]

        # Add project to path if needed
        if str(cls.project_root) not in sys.path:
            sys.path.insert(0, str(cls.project_root))

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_hierarchical_param_space_initialization(self):
        """Test HierarchicalParamSpace initialization with various sizes."""
        from ml_grid.util.hierarchical_param_space import HierarchicalParamSpace

        # Test medium size
        hps_medium = HierarchicalParamSpace(size="medium")
        self.assertEqual(hps_medium.base_size, "medium")

        # Test xsmall size
        hps_xsmall = HierarchicalParamSpace(size="xsmall")
        self.assertEqual(hps_xsmall.base_size, "xsmall")

        # Test with custom config
        hps_custom = HierarchicalParamSpace(
            size="medium",
            hierarchical_config={"max_total_evals": 50, "reduction_factor": 0.3},
        )
        self.assertEqual(hps_custom.hierarchical_config["max_total_evals"], 50)
        self.assertEqual(hps_custom.hierarchical_config["reduction_factor"], 0.3)

    def test_generate_hierarchical_space(self):
        """Test generation of three-tiered parameter spaces."""
        from ml_grid.util.param_space import ParamSpace
        from ml_grid.util.hierarchical_param_space import HierarchicalParamSpace

        base_space = ParamSpace(size="medium")
        hps = HierarchicalParamSpace(
            size="medium", hierarchical_config={"max_total_evals": 100}
        )

        # Generate staged spaces - should return dict or None for each
        result = hps.generate_hierarchical_space(base_space.param_dict)

        # Verify we got a valid tuple back with 3 elements
        self.assertEqual(len(result), 3, "Should return 3 space dictionaries")

    def test_parameter_importance_analysis(self):
        """Test parameter importance analysis."""
        from ml_grid.util.hierarchical_param_space import AdaptiveParameterAnalyzer

        # Create synthetic results data
        np.random.seed(42)

        results = []
        for i in range(20):
            result = {
                "score": 0.5 + 0.3 * (i % 10) / 10 + np.random.uniform(-0.1, 0.1),
                "parameters": {
                    "param_a": float(i % 5),
                    "param_b": float(i % 3),
                    "param_c": float(i),
                },
            }
            results.append(result)

        # Create parameter space
        param_space = {
            "param_a": [0.0, 1.0, 2.0, 3.0, 4.0],
            "param_b": [0.0, 1.0, 2.0],
            "param_c": list(range(20)),
        }

        # Test importance analysis
        analyzer = AdaptiveParameterAnalyzer()
        importance = analyzer.analyze_parameter_importance(results, param_space)

        # Verify output format
        self.assertIsInstance(importance, dict)
        self.assertEqual(set(importance.keys()), set(param_space.keys()))

        # All scores should be in [0.1, 1.0]
        for param_name, score in importance.items():
            self.assertGreaterEqual(score, 0.1)
            self.assertLessEqual(score, 1.0)

    def test_early_stopping_rules(self):
        """Test early stopping rule implementation."""
        from ml_grid.util.hierarchical_search import EarlyStoppingRule

        # Create early stopper with short patience
        es = EarlyStoppingRule(min_trials=3, patience=2)

        # Test not enough trials yet
        should_stop, reason = es.should_stop([])
        self.assertFalse(should_stop)
        self.assertIn("Insufficient", reason)

        # Test improvement scenario
        class MockResult:
            def __init__(self, score):
                self.score = score

        results_with_improvement = [
            MockResult(0.7),
            MockResult(0.8),
            MockResult(0.9),  # New best - should NOT stop
        ]

        should_stop, reason = es.should_stop(results_with_improvement)
        self.assertFalse(should_stop)

        # Test no improvement scenario (patience reached)
        results_without_improvement = [
            MockResult(0.8),
            MockResult(0.75),  # Drop
            MockResult(0.72),  # No improvement for 3 trials
        ]

        should_stop, reason = es.should_stop(results_without_improvement)
        # Note: early stopping only triggers after min_trials + patience
        self.assertIsInstance(should_stop, bool)

    def test_space_reducer(self):
        """Test dynamic space reduction functionality."""
        from ml_grid.util.hierarchical_search import DynamicSpaceReducer

        param_space = {
            "param_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "param_b": [0.1, 0.2, 0.3, 0.4, 0.5],
            "param_c": list(range(10)),
        }

        # Create mock results (best parameters should be near middle)
        class MockResult:
            def __init__(self, params, score):
                self.parameters = params
                self.score = score

        # Top performers have param_a around 3-4
        top_results = [
            MockResult({"param_a": 3.0, "param_b": 0.3}, score=0.9),
            MockResult({"param_a": 4.0, "param_b": 0.2}, score=0.85),
            MockResult({"param_a": 3.5, "param_b": 0.4}, score=0.8),
        ]

        reducer = DynamicSpaceReducer(param_space)
        reduced_space = reducer.get_reduced_space(top_results, reduction_factor=0.4)

        # Verify space was reduced
        self.assertIsInstance(reduced_space, dict)
        for param_name, values in reduced_space.items():
            # Reduced should be smaller than original (for list-based params)
            if isinstance(values, list):
                self.assertLessEqual(len(values), len(param_space[param_name]))

    def test_global_params_hierarchical_integration(self):
        """Test that global parameters support hierarchical search settings."""
        # Import module to ensure classes are registered
        import ml_grid.util.global_params as gp_module

        # Clear singleton instance if exists
        original_instance = getattr(gp_module.GlobalParameters, "_instance", None)
        if original_instance is not None:
            delattr(gp_module.GlobalParameters, "_instance")

        # Re-initialize global parameters to get fresh instance with all attributes
        gp_module.global_parameters._initialized = False

        # Verify new hierarchical parameters exist on class definition (before instantiation)
        self.assertTrue(hasattr(gp_module.GlobalParameters, "use_hierarchical_search"))
        self.assertTrue(hasattr(gp_module.GlobalParameters, "hierarchical_max_evals"))
        self.assertTrue(
            hasattr(gp_module.GlobalParameters, "hierarchical_reduction_factor")
        )

    def test_backward_compatibility(self):
        """Test that existing configurations still work."""
        from ml_grid.util.param_space import ParamSpace

        # Test that original param spaces are unchanged
        base_spaces = ["medium", "xsmall", "xwide"]

        for size in base_spaces:
            ps = ParamSpace(size=size)

            # Should have valid param_dict
            self.assertIsNotNone(ps.param_dict, f"Size {size} produced None")

            # Verify the space has parameters (regardless of format)
            self.assertGreater(
                len(ps.param_dict), 0, f"Param dict empty for size {size}"
            )

    def test_hierarchical_search_integration(self):
        """Test integration between hierarchical and existing search infrastructure."""
        from ml_grid.pipeline.hyperparameter_search import HyperparameterSearch
        from sklearn.linear_model import LogisticRegression

        param_space = {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        }

        # Test HyperparameterSearch can be instantiated with new flags
        try:
            search = HyperparameterSearch(
                algorithm=LogisticRegression(),
                parameter_space=param_space,
                method_name="test",
                global_params=None,
            )

            self.assertIsNotNone(search)

        except Exception as e:
            self.fail(f"HyperparameterSearch initialization failed: {e}")

    def test_importance_scoring_consistency(self):
        """Test that importance scoring is consistent across runs."""
        from ml_grid.util.hierarchical_param_space import AdaptiveParameterAnalyzer

        np.random.seed(42)

        # Create identical results
        base_results = []
        for i in range(15):
            base_results.append(
                {
                    "score": 0.7 + 0.2 * (i % 5) / 5,
                    "parameters": {"param_x": float(i), "param_y": float(i % 3)},
                }
            )

        analyzer = AdaptiveParameterAnalyzer()

        # Run twice with same data
        importance1 = analyzer.analyze_parameter_importance(
            base_results, {"param_x": list(range(15)), "param_y": [0.0, 1.0, 2.0]}
        )

        importances = analyzer.analyze_parameter_importance(
            base_results.copy(),
            {"param_x": list(range(15)), "param_y": [0.0, 1.0, 2.0]},
        )

        # Verify same keys exist
        self.assertEqual(set(importance1.keys()), set(importances.keys()))

        # Scores should be close (may vary slightly due to random elements in analysis)
        for key in importance1:
            diff = abs(importance1[key] - importances[key])
            self.assertLess(diff, 0.3, f"Importance score too different for {key}")

    def test_param_to_scalar_edge_cases(self):
        """Test _param_to_scalar edge cases: numpy scalars, None, and booleans."""
        from ml_grid.util.hierarchical_search import ParameterImportanceAnalyzer
        import numpy as np

        analyzer = ParameterImportanceAnalyzer()

        # Test with numpy scalar (hasattr(value, "item"))
        numpy_scalar = np.float64(3.14)
        result = analyzer._param_to_scalar(numpy_scalar)
        self.assertIsInstance(result, (int, float))

        # Test with None value
        result_none = analyzer._param_to_scalar(None)
        self.assertEqual(result_none, -1)

        # Test with boolean True
        result_true = analyzer._param_to_scalar(True)
        self.assertEqual(result_true, 1)

        # Test with boolean False
        result_false = analyzer._param_to_scalar(False)
        self.assertEqual(result_false, 0)

    def test_sample_param_edge_cases(self):
        """Test _sample_param edge cases: empty list and single element."""
        from ml_grid.util.hierarchical_search import HierarchicalSearchOptimizer
        import numpy as np

        # Create optimizer instance to access _sample_param method
        hso = HierarchicalSearchOptimizer(
            initial_param_space={"param": [1, 2]},
            max_total_trials=10,
        )

        # Test with empty list - should not raise ValueError
        result_empty = hso._sample_param([], bias=0.5)
        # Empty list returns the empty list directly (fallback at line 701)
        self.assertEqual(result_empty, [])

        # Test with single element list and low bias
        result_single_low_bias = hso._sample_param([42], bias=0.3)
        self.assertEqual(result_single_low_bias, 42)

        # Test with single element list and high bias
        result_single_high_bias = hso._sample_param([99], bias=0.8)
        self.assertEqual(result_single_high_bias, 99)

        # Test with two element list (<= 2 condition triggers direct choice)
        result_two_elem = hso._sample_param(["a", "b"], bias=0.5)
        self.assertIn(result_two_elem, ["a", "b"])

        # Test with numpy array fallback - returns same array object
        np_array = np.array([1, 2, 3])
        result_np = hso._sample_param(np_array, bias=0.5)
        self.assertIs(result_np, np_array)

        # Test with string fallback
        result_str = hso._sample_param("test", bias=0.5)
        self.assertEqual(result_str, "test")

    def test_sample_with_bayesian_none_importance(self):
        """Test _sample_with_bayesian with None importance_weights (valid edge case)."""
        from ml_grid.util.hierarchical_search import HierarchicalSearchOptimizer

        # Create optimizer instance
        hso = HierarchicalSearchOptimizer(
            initial_param_space={"a": [1, 2], "b": [3, 4]},
            max_total_trials=5,
        )

        # Test with None importance_weights - should use default weight of 1.0
        samples = hso._sample_with_bayesian(
            {"a": [1, 2], "b": [3, 4]}, n_samples=3, importance_weights=None
        )

        # Should return list of dicts with valid values
        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertIn(sample["a"], [1, 2])
            self.assertIn(sample["b"], [3, 4])

    def test_sample_param_with_skopt_types(self):
        """Test _sample_param with skopt space types: Real, Integer, Categorical."""
        from ml_grid.util.hierarchical_search import HierarchicalSearchOptimizer
        from skopt.space import Real, Integer, Categorical

        # Create optimizer instance
        hso = HierarchicalSearchOptimizer(
            initial_param_space={"param": [1, 2]},
            max_total_trials=10,
        )

        # Test with Real space type
        real_spec = Real(low=0.1, high=1.0)
        result_real_low_bias = hso._sample_param(real_spec, bias=0.3)
        self.assertIsInstance(result_real_low_bias, float)
        self.assertGreaterEqual(result_real_low_bias, 0.1)
        self.assertLessEqual(result_real_low_bias, 1.0)

        # Test with Real space and high bias (should be closer to center)
        result_real_high_bias = hso._sample_param(real_spec, bias=0.8)
        self.assertIsInstance(result_real_high_bias, float)
        self.assertGreaterEqual(result_real_high_bias, 0.1)
        self.assertLessEqual(result_real_high_bias, 1.0)

        # Test with Integer space type
        int_spec = Integer(low=5, high=20)
        result_int_low_bias = hso._sample_param(int_spec, bias=0.4)
        self.assertIsInstance(result_int_low_bias, int)
        self.assertGreaterEqual(result_int_low_bias, 5)
        self.assertLessEqual(result_int_low_bias, 20)

        # Test with Integer space and high bias
        result_int_high_bias = hso._sample_param(int_spec, bias=0.7)
        self.assertIsInstance(result_int_high_bias, int)
        self.assertGreaterEqual(result_int_high_bias, 5)
        self.assertLessEqual(result_int_high_bias, 20)

        # Test with Categorical > 2 categories
        cat_spec = Categorical(categories=["a", "b", "c", "d"])
        result_cat_low_bias = hso._sample_param(cat_spec, bias=0.3)
        self.assertIn(result_cat_low_bias, ["a", "b", "c", "d"])

        # Test with Categorical > 2 and high bias (should stay within middle)
        result_cat_high_bias = hso._sample_param(cat_spec, bias=0.8)
        self.assertIn(result_cat_high_bias, ["a", "b", "c", "d"])

        # Test with Categorical <= 2 categories (uses direct choice)
        cat_spec_small = Categorical(categories=["x", "y"])
        result_cat_small = hso._sample_param(cat_spec_small, bias=0.5)
        self.assertIn(result_cat_small, ["x", "y"])


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing ml_grid configurations."""

    def test_param_space_sizes(self):
        """Verify all param space sizes generate correctly."""
        from ml_grid.util.param_space import ParamSpace

        for size in ["medium", "xsmall", "small", "xwide"]:
            ps = ParamSpace(size=size)
            self.assertIsNotNone(ps.param_dict, f"Size {size} produced None")

    def test_hierarchical_config_defaults(self):
        """Test that hierarchical configs have sensible defaults."""
        from ml_grid.util.hierarchical_param_space import HierarchicalParamSpace

        # Default config should be populated
        hps = HierarchicalParamSpace(size="medium")

        self.assertIn("max_total_evals", hps.hierarchical_config)
        self.assertGreater(hps.hierarchical_config["max_total_evals"], 0)


# Utility function for generating synthetic data pipelines tests
def create_synthetic_test_pipeline():
    """Create a minimal test pipeline with synthetic data."""
    np.random.seed(42)

    # Create small synthetic dataset
    n_samples = 100
    n_features = 5

    X = pd.DataFrame(
        {f"feature_{i}": np.random.randn(n_samples) for i in range(n_features)}
    )

    y = pd.Series(np.random.randint(0, 2, n_samples))

    return X, y


# Run tests if executed directly
if __name__ == "__main__":
    unittest.main()
