"""Test HierarchicalHyperparameterSearch integration."""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from unittest.mock import Mock

import sys

# Add project to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestHierarchicalHyperparameterSearch(unittest.TestCase):
    """Test HierarchicalHyperparameterSearch integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic data
        np.random.seed(42)
        self.X_train = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "feature_3": np.random.randn(100),
            }
        )
        self.y_train = pd.Series(np.random.randint(0, 2, 100))

    def test_init(self):
        """Test HierarchicalHyperparameterSearch initialization."""
        from ml_grid.pipeline.hierarchical_hyperparameter_search import (
            HierarchicalHyperparameterSearch,
        )

        algo = LogisticRegression()
        param_space = {"C": [0.1, 1.0], "penalty": ["l1", "l2"]}

        search = HierarchicalHyperparameterSearch(
            algorithm=algo,
            parameter_space=param_space,
            method_name="TestModel",
            global_params=None,  # Will use default
            ml_grid_object=self._create_mock_ml_grid(),
        )

        self.assertEqual(search.method_name, "TestModel")
        self.assertIsNotNone(search.parameter_space)

    def test_run_hierarchical_search_full(self):
        """Test full hierarchical search execution."""
        from ml_grid.pipeline.hierarchical_hyperparameter_search import (
            HierarchicalHyperparameterSearch,
        )
        from ml_grid.util.global_params import global_parameters

        algo = LogisticRegression(max_iter=100)
        param_space = {"C": [0.1, 1.0, 10.0], "penalty": ["l1", "l2"]}

        search = HierarchicalHyperparameterSearch(
            algorithm=algo,
            parameter_space=param_space,
            method_name="TestModel",
            global_params=global_parameters,
            ml_grid_object=self._create_mock_ml_grid(),
            max_total_evals=10,  # Small for testing
        )

        best_estimator, all_results = search.run_hierarchical_search(
            self.X_train, self.y_train
        )

        # Check results structure
        self.assertIsNotNone(all_results)
        self.assertIn("coarse", all_results)
        self.assertIn("fine", all_results)
        self.assertIn("refinement", all_results)

    def test_preprocess_data(self):
        """Test data preprocessing."""
        from ml_grid.pipeline.hierarchical_hyperparameter_search import (
            HierarchicalHyperparameterSearch,
        )
        from sklearn.linear_model import LogisticRegression

        search = HierarchicalHyperparameterSearch(
            algorithm=LogisticRegression(),
            parameter_space={"C": [0.1, 1.0]},
            method_name="Test",
            global_params=None,
            ml_grid_object=self._create_mock_ml_grid(),
        )

        X_prep, y_prep = search._preprocess_data(self.X_train, self.y_train)

        # Check preprocessing
        self.assertIsInstance(X_prep.columns[0], str)  # Columns should be strings

    def test_build_evaluation_function(self):
        """Test evaluation function building."""
        from ml_grid.pipeline.hierarchical_hyperparameter_search import (
            HierarchicalHyperparameterSearch,
        )
        from sklearn.linear_model import LogisticRegression

        search = HierarchicalHyperparameterSearch(
            algorithm=LogisticRegression(),
            parameter_space={"C": [0.1, 1.0]},
            method_name="Test",
            global_params=None,
            ml_grid_object=self._create_mock_ml_grid(),
        )

        eval_fn = search._build_evaluation_function(self.X_train, self.y_train)

        # Test evaluation function
        test_params = {"C": 1.0}
        score, fit_time = eval_fn(test_params)

        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertGreater(fit_time, 0)

    def _create_mock_ml_grid(self):
        """Create mock ml_grid object."""
        mock_obj = Mock()
        mock_obj.X_train_orig = self.X_train.copy()
        mock_obj.y_train_orig = self.y_train.copy()
        return mock_obj


if __name__ == "__main__":
    unittest.main()
