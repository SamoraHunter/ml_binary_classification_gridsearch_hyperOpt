import unittest
import shutil
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure the project root is in sys.path so we can import the module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_grid.util.project_score_save import project_score_save_class


class TestProjectScoreSave(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for the experiment to avoid cluttering disk
        self.test_dir = tempfile.mkdtemp()
        self.experiment_dir = Path(self.test_dir) / "test_experiment"

        # Patch global_parameters to control configuration during tests
        self.patcher = patch("ml_grid.util.project_score_save.global_parameters")
        self.mock_globals = self.patcher.start()

        # Default mock configuration
        self.mock_globals.metric_list = {"auc": "auc", "accuracy": "accuracy"}
        self.mock_globals.error_raise = (
            True  # Important: Raise errors so tests fail on bugs
        )
        self.mock_globals.bayessearch = False
        self.mock_globals.store_models = False

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test that the log file is created with correct headers."""
        project_score_save_class(str(self.experiment_dir))

        log_path = self.experiment_dir / "final_grid_score_log.csv"
        self.assertTrue(log_path.exists(), "Log file was not created")

        df = pd.read_csv(log_path)
        expected_cols = ["algorithm_implementation", "auc_m", "accuracy_m"]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_update_score_log_success(self):
        """Test a successful write to the log file with all attributes present."""
        saver = project_score_save_class(str(self.experiment_dir))

        # Mock the ml_grid_object with all expected attributes
        mock_grid = MagicMock()
        mock_grid.X_train = [1, 2]
        mock_grid.y_train = [0, 1]
        mock_grid.X_test = pd.DataFrame({"a": [1]})
        mock_grid.y_test = pd.Series([1, 0])
        mock_grid.X_test_orig = [1, 2]
        mock_grid.y_test_orig = [1, 0]
        mock_grid.param_space_index = 1
        mock_grid.outcome_variable = "target"

        # Attributes that caused issues previously
        mock_grid.local_param_dict = {"param1": 10}
        mock_grid.final_column_list = ["col1"]
        mock_grid.original_feature_names = ["col1", "col2"]

        # Mock scores and algorithm
        scores = {
            "fit_time": [0.1],
            "score_time": [0.01],
            "test_auc": [0.8],
            "test_accuracy": [0.9],
        }
        best_pred = np.array([1, 0])
        algo = MagicMock()
        algo.get_params.return_value = {"p": 1}

        saver.update_score_log(
            ml_grid_object=mock_grid,
            scores=scores,
            best_pred_orig=best_pred,
            current_algorithm=algo,
            method_name="TestAlgo",
            pg=10,
            start=0,
            n_iter_v=5,
            failed=False,
        )

        # Verify data was written
        log_path = self.experiment_dir / "final_grid_score_log.csv"
        df = pd.read_csv(log_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["method_name"], "TestAlgo")
        self.assertEqual(df.iloc[0]["auc_m"], 0.8)

    def test_update_score_log_typo_and_missing_safety(self):
        """Test that the code handles missing attributes and the 'orignal' typo."""
        saver = project_score_save_class(str(self.experiment_dir))

        mock_grid = MagicMock()
        # Minimal setup
        mock_grid.y_test = pd.Series([1, 0])
        mock_grid.param_space_index = 1

        # Simulate missing local_param_dict (should default to {})
        del mock_grid.local_param_dict

        # Simulate the typo: 'original' missing, 'orignal' present
        del mock_grid.original_feature_names
        mock_grid.orignal_feature_names = ["col1"]
        mock_grid.final_column_list = ["col1"]

        scores = {
            "fit_time": [0.1],
            "score_time": [0.01],
            "test_auc": [0.5],
            "test_accuracy": [0.5],
        }

        # Should not raise AttributeError
        saver.update_score_log(
            ml_grid_object=mock_grid,
            scores=scores,
            best_pred_orig=np.array([1, 0]),
            current_algorithm=MagicMock(),
            method_name="TypoTest",
            pg=1,
            start=0,
            n_iter_v=1,
            failed=False,
        )

        log_path = self.experiment_dir / "final_grid_score_log.csv"
        df = pd.read_csv(log_path)
        self.assertEqual(len(df), 1)

    def test_initialization_does_not_overwrite(self):
        """Test that re-initializing the class does not wipe an existing log file."""
        # First initialization
        project_score_save_class(str(self.experiment_dir))
        log_path = self.experiment_dir / "final_grid_score_log.csv"

        # Simulate writing some data
        with open(log_path, "a") as f:
            f.write("test_data_entry\n")

        # Second initialization on same directory
        project_score_save_class(str(self.experiment_dir))

        # Verify data persists
        with open(log_path, "r") as f:
            content = f.read()
        self.assertIn("test_data_entry", content)


if __name__ == "__main__":
    unittest.main()
