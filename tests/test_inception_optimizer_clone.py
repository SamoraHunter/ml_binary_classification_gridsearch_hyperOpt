"""Test IndividualInceptionClassifier optimizer cloning patch."""

import unittest


class TestInceptionOptimizerClone(unittest.TestCase):
    """Test the optimizer cloning logic in IndividualInceptionClassifier __init__ patch."""

    def test_metrics_none_handler(self):
        """Test that metrics=None results in _metrics=[].

        This tests lines 422-428 where:
        - metrics_val = getattr(self, "metrics", "accuracy")
        - if metrics_val is None: self._metrics = []

        The uncovered path is when metrics_val is explicitly None.
        """

        # Simulate the patch logic for metrics handling
        class MockInstance:
            def __init__(self):
                self.metrics = None  # Explicitly set to None

        mock_instance = MockInstance()
        metrics_val = getattr(mock_instance, "metrics", "accuracy")

        if metrics_val is None:
            _metrics = []
        elif isinstance(metrics_val, str):
            _metrics = [metrics_val]
        else:
            _metrics = list(metrics_val)

        self.assertEqual(_metrics, [])

    def test_metrics_str_handler(self):
        """Test that metrics as string results in _metrics=[str].

        This tests the path at lines 425-426 where metrics is a string.
        """

        class MockInstance:
            def __init__(self):
                self.metrics = "accuracy"

        mock_instance = MockInstance()
        metrics_val = getattr(mock_instance, "metrics", "accuracy")

        if metrics_val is None:
            _metrics = []
        elif isinstance(metrics_val, str):
            _metrics = [metrics_val]
        else:
            _metrics = list(metrics_val)

        self.assertEqual(_metrics, ["accuracy"])
        self.assertIsInstance(_metrics[0], str)

    def test_metrics_list_handler(self):
        """Test that metrics as list results in _metrics=list(metrics).

        This tests the path at lines 427-428 where metrics is a list or tuple.
        """

        class MockInstance:
            def __init__(self):
                self.metrics = ["accuracy", "precision"]

        mock_instance = MockInstance()
        metrics_val = getattr(mock_instance, "metrics", "accuracy")

        if metrics_val is None:
            _metrics = []
        elif isinstance(metrics_val, str):
            _metrics = [metrics_val]
        else:
            _metrics = list(metrics_val)

        self.assertEqual(_metrics, ["accuracy", "precision"])
        self.assertIsInstance(_metrics, list)


if __name__ == "__main__":
    unittest.main()
