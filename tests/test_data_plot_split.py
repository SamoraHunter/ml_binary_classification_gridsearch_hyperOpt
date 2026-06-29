"""Tests for data_plot_split module."""

import unittest


class TestPlotPieChartWithCounts(unittest.TestCase):
    """Test plot_pie_chart_with_counts function."""

    def test_empty_datasets_warning(self):
        """Test that empty datasets trigger a warning and early return."""

        with self.assertLogs("ml_grid", level="WARNING") as log:
            from ml_grid.pipeline.data_plot_split import plot_pie_chart_with_counts

            plot_pie_chart_with_counts([], [], [])

        self.assertTrue(
            any("Cannot plot pie chart" in msg for msg in log.output),
            "Warning message not logged",
        )

    def test_non_empty_datasets(self):
        """Test that non-empty datasets don't trigger warnings."""
        import logging
        from io import StringIO

        # Capture logging output
        logger = logging.getLogger("ml_grid")
        old_handler = logger.handlers[:]
        logger.handlers.clear()

        string_io = StringIO()
        handler = logging.StreamHandler(string_io)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)

        from ml_grid.pipeline.data_plot_split import plot_pie_chart_with_counts

        # This should not produce any warnings
        plot_pie_chart_with_counts([1, 2], [3, 4], [5, 6])

        log_output = string_io.getvalue()
        self.assertNotIn(
            "Cannot plot pie chart",
            log_output,
            "Should not warn for non-empty datasets",
        )

        # Restore handlers
        logger.handlers.clear()
        logger.handlers.extend(old_handler)


class TestCreateBarChart(unittest.TestCase):
    """Test create_bar_chart function."""

    def test_simple_dict(self):
        """Test basic bar chart creation from a dictionary."""
        from ml_grid.pipeline.data_plot_split import create_bar_chart

        data = {"A": 10, "B": 20, "C": 30}

        result = create_bar_chart(data, title="Test", x_label="X", y_label="Y")

        self.assertIsNone(result)

    def test_empty_dict(self):
        """Test bar chart with empty dictionary."""
        from ml_grid.pipeline.data_plot_split import create_bar_chart

        data = {}

        result = create_bar_chart(data)

        self.assertIsNone(result)

    def test_single_item_dict(self):
        """Test bar chart with single item."""
        from ml_grid.pipeline.data_plot_split import create_bar_chart

        data = {"A": 10}

        result = create_bar_chart(data)

        self.assertIsNone(result)


class TestPlotDictValues(unittest.TestCase):
    """Test plot_dict_values function."""

    def test_bool_dict_true_false(self):
        """Test plotting a dictionary with boolean values."""
        from ml_grid.pipeline.data_plot_split import plot_dict_values

        data = {"field1": True, "field2": False, "field3": True}

        result = plot_dict_values(data)

        self.assertIsNone(result)

    def test_empty_bool_dict(self):
        """Test plotting an empty dictionary."""
        from ml_grid.pipeline.data_plot_split import plot_dict_values

        data = {}

        result = plot_dict_values(data)

        self.assertIsNone(result)


class TestPlotCandidateFeatureCategoryLists(unittest.TestCase):
    """Test plot_candidate_feature_category_lists function."""

    def test_simple_feature_counts(self):
        """Test plotting feature category counts."""
        from ml_grid.pipeline.data_plot_split import (
            plot_candidate_feature_category_lists,
        )

        data = {"category1": 5, "category2": 10, "category3": 3}

        result = plot_candidate_feature_category_lists(data)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
