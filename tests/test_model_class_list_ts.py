"""
Unit tests for ml_grid.pipeline.model_class_list_ts module.

This test suite validates the time-series model list generation functionality,
ensuring that models are correctly instantiated based on configuration.
"""

import unittest

try:
    from ml_grid.pipeline.model_class_list_ts import get_model_class_list_ts
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from ml_grid.pipeline.model_class_list_ts import get_model_class_list_ts
    except ImportError:
        get_model_class_list_ts = None


if get_model_class_list_ts is None:
    raise unittest.SkipTest(
        "aeon module not installed - skipping time-series model tests"
    )


class MockLogger:
    """Mock logger for testing without actual logging output."""

    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(("info", msg))

    def warning(self, msg):
        self.messages.append(("warning", msg))

    def debug(self, msg):
        self.messages.append(("debug", msg))

    def error(self, msg, exc_info=False):
        self.messages.append(("error", msg))

    def critical(self, msg):
        self.messages.append(("critical", msg))


class MockPipe:
    """Mock pipe object for testing get_model_class_list_ts."""

    def __init__(self, model_class_dict=None):
        self.model_class_dict = model_class_dict
        self.logger = MockLogger()


class TestGetModelClassListTs(unittest.TestCase):
    """Test suite for the get_model_class_list_ts function."""

    def test_get_model_class_list_ts_none_dict_returns_empty_list(self):
        """
        Tests that when model_class_dict is None, an empty list is returned.

        This covers the edge case where no time-series models are configured.
        """
        mock_pipe = MockPipe(model_class_dict=None)

        result = get_model_class_list_ts(mock_pipe)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)
        warning_found = any(
            msg[0] == "warning" and "model_class_dict is None" in msg[1]
            for msg in mock_pipe.logger.messages
        )
        self.assertTrue(
            warning_found,
            "Expected warning message about model_class_dict being None not found",
        )

    def test_get_model_class_list_ts_empty_dict_returns_empty_list(self):
        """
        Tests that when model_class_dict is an empty dict, an empty list is returned.
        """
        mock_pipe = MockPipe(model_class_dict={})

        result = get_model_class_list_ts(mock_pipe)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_get_model_class_list_ts_all_false_returns_empty_list(self):
        """
        Tests that when all model includes are False, an empty list is returned.
        """
        mock_pipe = MockPipe(
            model_class_dict={
                "KNeighborsTimeSeriesClassifier": False,
                "TimeSeriesForestClassifier": False,
            }
        )

        result = get_model_class_list_ts(mock_pipe)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_get_model_class_list_ts_single_include(self):
        """
        Tests that when a single model is set to True, it is instantiated.
        """
        mock_pipe = MockPipe(
            model_class_dict={
                "KNeighborsTimeSeriesClassifier": True,
            }
        )

        result = get_model_class_list_ts(mock_pipe)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
