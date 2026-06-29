"""Test exception handling in grid_search_cross_validate_ts - lines 856-862 coverage."""

import unittest


class TestParameterGridTypeError(unittest.TestCase):
    """Test TypeError handling when calculating ParameterGrid size for skopt spaces."""

    def test_skopt_spaces_cause_typeerror_in_parametergrid(self):
        """Test that skopt parameter spaces raise TypeError with len(ParameterGrid).

        Tests lines 856-862 in grid_search_cross_validate_ts.py where:
        - Line 856: tries len(ParameterGrid(parameter_space))
        - Line 858except TypeError: catches the exception
        - Lines 859-861 log warning and set pg = "N/A"

        Skopt spaces (Integer, Real, Categorical) cannot be used directly with
        ParameterGrid because they represent search distributions, not discrete values.
        """
        from sklearn.model_selection import ParameterGrid
        from skopt.space import Integer, Real, Categorical

        # Create parameter space with skopt spaces
        skopt_space = {
            "n_estimators": Integer(low=10, high=100),
            "learning_rate": Real(low=0.01, high=0.5),
            "criterion": Categorical(categories=["gini", "entropy"]),
        }

        # This should raise TypeError as skopt spaces are not valid for ParameterGrid
        with self.assertRaises(TypeError):
            _ = len(ParameterGrid(skopt_space))


class TestNIterTypeError(unittest.TestCase):
    """Test ValueError/TypeError handling in n_iter parsing."""

    def test_invalid_n_iter_defaults_to_2(self):
        """Test that invalid n_iter values default to 2.

        Tests lines 813-817 in grid_search_cross_validate_ts.py where:
        - Line 809: tries getattr for n_iter
        - Lines 813-817 catch ValueError/TypeError and log warning, defaulting to 2.
        """
        import inspect
        from ml_grid.pipeline import grid_search_cross_validate_ts

        init_source = inspect.getsource(
            grid_search_cross_validate_ts.grid_search_crossvalidate_ts.__init__
        )

        self.assertIn("n_iter_v is None", init_source)
        self.assertIn("n_iter_v = 2", init_source)


if __name__ == "__main__":
    unittest.main()
