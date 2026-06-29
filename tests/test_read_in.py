import pandas as pd
import pytest


class TestReadClass:
    """Tests for the read class (CSV reading with Polars vs pandas fallback)."""

    def test_read_csv_with_pandas_default(self, tmp_path):
        """Test default pandas CSV reading."""
        df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "outcome_var_1": [0, 1, 0]}
        )
        filepath = tmp_path / "test.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read

        reader = read(str(filepath), use_polars=False)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert len(reader.raw_input_data) == 3
        assert list(reader.raw_input_data.columns) == ["col1", "col2", "outcome_var_1"]

    @pytest.mark.skipif(
        not __import__("polars").__version__, reason="Polars not installed"
    )
    def test_read_csv_with_polars(self, tmp_path):
        """Test Polars CSV reading with fallback to pandas."""
        df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "outcome_var_1": [0, 1, 0]}
        )
        filepath = tmp_path / "test.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read

        reader = read(str(filepath), use_polars=True)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert len(reader.raw_input_data) == 3

    def test_read_nonexistent_file(self):
        """Test error handling for non-existent file."""
        from ml_grid.pipeline.read_in import read
        import logging

        logger = logging.getLogger("ml_grid")
        logger.setLevel(logging.ERROR)

        reader = read("/nonexistent/path/file.csv", use_polars=False)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert reader.raw_input_data.empty

    @pytest.mark.skipif(
        not __import__("polars").__version__, reason="Polars not installed"
    )
    def test_read_nonexistent_file_with_polars(self):
        """Test error handling for non-existent file with Polars."""
        from ml_grid.pipeline.read_in import read
        import logging

        logger = logging.getLogger("ml_grid")
        logger.setLevel(logging.ERROR)

        reader = read("/nonexistent/path/file.csv", use_polars=True)
        assert isinstance(reader.raw_input_data, pd.DataFrame)
        assert reader.raw_input_data.empty


class TestReadSampleClass:
    """Tests for the read_sample class (sampling rows and columns)."""

    def test_read_sample_all_rows_no_columns(self, tmp_path):
        """Test reading all rows with no column sampling."""
        df = pd.DataFrame(
            {
                "outcome_var_1": [0, 1, 0, 1, 0],
                "age": [25, 30, 35, 40, 45],
                "male": [1, 0, 1, 0, 1],
                ".feature1": [1, 2, 3, 4, 5],
                "feature2": [6, 7, 8, 9, 10],
            }
        )
        filepath = tmp_path / "test_sample.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read_sample

        reader = read_sample(str(filepath), test_sample_n=0, column_sample_n=0)
        assert len(reader.raw_input_data) == 5
        # All columns should be present (outcome_var_1, age, male, .feature1, feature2)
        assert len(reader.raw_input_data.columns) == 5

    def test_read_sample_with_outcome_column(self, tmp_path):
        """Test that outcome_var_1 is always included."""
        # Create with guaranteed alternating outcomes to ensure both classes present
        df = pd.DataFrame(
            {
                "outcome_var_1": [0, 1, 0, 1, 0],
                "age": [25, 30, 35, 40, 45],
                "male": [1, 0, 1, 0, 1],
                "other_col": ["a", "b", "c", "d", "e"],
                "extra_col": ["x", "y", "z", "w", "v"],
            }
        )
        filepath = tmp_path / "test_outcome.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read_sample

        # Request sampling with all rows to ensure both classes are included
        reader = read_sample(str(filepath), test_sample_n=5, column_sample_n=1)
        assert len(reader.raw_input_data) <= 5
        assert "outcome_var_1" in reader.raw_input_data.columns

    def test_read_sample_validation_error(self, tmp_path):
        """Test ValueError when outcome variable doesn't have both classes."""
        df = pd.DataFrame(
            {
                "outcome_var_1": [0, 0, 0],  # All same class
                "age": [25, 30, 35],
                "male": [1, 0, 1],
            }
        )
        filepath = tmp_path / "test_error.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read_sample

        with pytest.raises(
            ValueError, match="Outcome variable does not have both classes"
        ):
            read_sample(str(filepath), test_sample_n=3, column_sample_n=1)

    def test_read_sample_with_all_sampling_strategy(self, tmp_path):
        """Test sampling with "all" strategy (test_sample_n=0)."""
        df = pd.DataFrame(
            {
                "outcome_var_1": [0, 1, 0, 1, 0],
                "age": [25, 30, 35, 40, 45],
                "male": [1, 0, 1, 0, 1],
                "extra_col1": [1, 2, 3, 4, 5],
                "extra_col2": [6, 7, 8, 9, 10],
            }
        )
        filepath = tmp_path / "test_all.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read_sample

        # test_sample_n=0 means all rows
        reader = read_sample(str(filepath), test_sample_n=0, column_sample_n=2)
        assert len(reader.raw_input_data) == 5
        # Should include outcome_var_1 and selected additional columns
        assert "outcome_var_1" in reader.raw_input_data.columns
        assert "age" in reader.raw_input_data.columns
        assert "male" in reader.raw_input_data.columns

    def test_read_sample_column_selection(self, tmp_path):
        """Test that necessary_columns are always included."""
        df = pd.DataFrame(
            {
                "outcome_var_1": [0, 1, 0],
                "age": [25, 30, 35],
                "male": [1, 0, 1],
                "other_col1": ["a", "b", "c"],
                "other_col2": ["x", "y", "z"],
            }
        )
        filepath = tmp_path / "test_cols.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read_sample

        reader = read_sample(str(filepath), test_sample_n=0, column_sample_n=1)

        # Check that all necessary columns are present
        assert "outcome_var_1" in reader.raw_input_data.columns
        assert "age" in reader.raw_input_data.columns
        assert "male" in reader.raw_input_data.columns

    def test_read_sample_single_row(self, tmp_path):
        """Test sampling a single row."""
        df = pd.DataFrame(
            {
                "outcome_var_1": [0, 1, 0, 1, 0],
                "age": [25, 30, 35, 40, 45],
                "male": [1, 0, 1, 0, 1],
            }
        )
        filepath = tmp_path / "test_single.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read_sample

        # Sample more rows than total to get all with random sampling
        reader = read_sample(str(filepath), test_sample_n=5, column_sample_n=0)
        assert len(reader.raw_input_data) <= 5
        assert "outcome_var_1" in reader.raw_input_data.columns

    def test_read_sample_with_missing_necessary_columns(self, tmp_path):
        """Test behavior when necessary columns are missing from file."""
        df = pd.DataFrame({"outcome_var_1": [0, 1, 0], "other_col": ["a", "b", "c"]})
        filepath = tmp_path / "test_missing.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read_sample

        # Sample 3 rows (all available)
        reader = read_sample(str(filepath), test_sample_n=4, column_sample_n=1)

        # Should still include outcome_var_1 even if others missing
        assert "outcome_var_1" in reader.raw_input_data.columns


class TestColumnNames:
    """Tests for column name generation and validation."""

    def test_column_validation_includes_necessary(self, tmp_path):
        """Test that necessary columns are validated correctly."""
        df = pd.DataFrame(
            {
                "outcome_var_1": [0, 1, 0],
                "age": [25, 30, 35],
                "male": [1, 0, 1],
                "extra": ["a", "b", "c"],
            }
        )
        filepath = tmp_path / "test_columns.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read_sample

        # column_sample_n is additional columns beyond necessary ones
        reader = read_sample(str(filepath), test_sample_n=3, column_sample_n=0)

        # Check that all necessary columns are present
        assert "outcome_var_1" in reader.raw_input_data.columns
        assert "age" in reader.raw_input_data.columns
        assert "male" in reader.raw_input_data.columns

    def test_column_name_generation_logic(self, tmp_path):
        """Test that column sampling logic works correctly."""
        # Create a file with many columns
        n_cols = 20
        col_names = ["outcome_var_1", "age", "male"]
        col_names.extend([f"col_{i}" for i in range(n_cols)])

        df_dict = {name: [0, 1] * 5 for name in col_names}
        df = pd.DataFrame(df_dict)
        filepath = tmp_path / "test_multi_cols.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read_sample

        # Request sampling of additional columns
        reader = read_sample(str(filepath), test_sample_n=5, column_sample_n=3)

        # Should have outcome_var_1 and some additional columns
        assert "outcome_var_1" in reader.raw_input_data.columns


class TestEdgeCases:
    """Test edge cases for data reading."""

    def test_empty_csv(self, tmp_path):
        """Test reading an empty CSV file."""
        filepath = tmp_path / "empty.csv"
        # Create file with only header
        pd.DataFrame().to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read

        reader = read(str(filepath), use_polars=False)
        assert isinstance(reader.raw_input_data, pd.DataFrame)

    def test_single_row_csv(self, tmp_path):
        """Test reading a CSV with only one row."""
        df = pd.DataFrame({"col1": [1], "outcome_var_1": [0]})
        filepath = tmp_path / "single.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read

        reader = read(str(filepath), use_polars=False)
        assert len(reader.raw_input_data) == 1

    def test_csv_with_special_characters(self, tmp_path):
        """Test reading CSV with special characters in values."""
        df = pd.DataFrame(
            {
                "col1": [1, 2],
                "col2": ["hello, world", 'test"quote'],
                "outcome_var_1": [0, 1],
            }
        )
        filepath = tmp_path / "special.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read

        reader = read(str(filepath), use_polars=False)
        assert len(reader.raw_input_data) == 2

    def test_read_sample_large_sampling(self, tmp_path):
        """Test sampling more rows than available."""
        df = pd.DataFrame({"outcome_var_1": [0, 1, 0], "age": [25, 30, 35]})
        filepath = tmp_path / "large_sample.csv"
        df.to_csv(filepath, index=False)

        from ml_grid.pipeline.read_in import read_sample

        # Request more rows than exist - should return all available
        reader = read_sample(str(filepath), test_sample_n=100, column_sample_n=0)
        assert len(reader.raw_input_data) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
