"""
Unit tests for data_percent_missing module.

Tests the handle_percent_missing function which identifies columns with high
percentages of missing data.
"""

from pathlib import Path

try:
    from ml_grid.pipeline.data_percent_missing import handle_percent_missing
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ml_grid.pipeline.data_percent_missing import handle_percent_missing


class TestHandlePercentMissing:
    """Test suite for handle_percent_missing function."""

    def test_file_not_found_logs_warning_and_returns_empty_drop_list(self):
        """Test behavior when percent missing file doesn't exist."""
        local_param_dict = {"percent_missing": 50}
        all_df_columns = ["col1", "col2", "col3"]
        file_name = "non_existent_file.csv"
        drop_list = []

        result = handle_percent_missing(
            local_param_dict, all_df_columns, file_name, drop_list
        )

        assert result == []

    def test_none_threshold_returns_drop_list_unmodified(self):
        """Test that None threshold doesn't modify drop list."""
        local_param_dict = {"percent_missing": None}
        all_df_columns = ["col1", "col2"]
        file_name = "test_file.csv"
        drop_list = ["existing_col"]

        result = handle_percent_missing(
            local_param_dict, all_df_columns, file_name, drop_list
        )

        assert result == ["existing_col"]

    def test_empty_dict_returns_drop_list_unmodified(self):
        """Test that empty percent missing dict doesn't modify drop list."""
        local_param_dict = {"percent_missing": 50}
        all_df_columns = ["col1", "col2"]
        file_name = "test_file.csv"
        drop_list = ["existing_col"]

        result = handle_percent_missing(
            local_param_dict, all_df_columns, file_name, drop_list
        )

        assert result == ["existing_col"]
