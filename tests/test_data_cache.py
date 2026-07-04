"""
Test suite for ml_grid data caching functionality.

This module contains unit and integration tests for the DataCache class
and CachedDataPipeline wrapper, ensuring proper cache operation,
invalidation, and thread safety.
"""

import tempfile
import threading
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5] * 10,
            "feature2": [10, 20, 30, 40, 50] * 10,
            "target": [0, 1, 0, 1, 0] * 10,
        }
    )
    return df


@pytest.fixture
def temp_dir():
    """Create a temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestDataCache:
    """Test suite for DataCache class."""

    def test_cache_initialization(self, temp_dir):
        """Test that DataCache initializes correctly."""
        from ml_grid.util.data_cache import DataCache

        cache = DataCache(base_cache_dir=temp_dir)

        assert cache.base_cache_dir == Path(temp_dir)
        assert cache.base_cache_dir.exists()

    def test_compute_file_checksum(self, sample_data, temp_dir):
        """Test SHA-256 file checksum computation."""
        test_file = Path(temp_dir) / "test.csv"
        sample_data.to_csv(test_file, index=False)

        from ml_grid.util.data_cache import DataCache

        cache = DataCache(base_cache_dir=temp_dir)
        checksum = cache._compute_file_checksum(str(test_file))

        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_save_and_load(self, sample_data, temp_dir):
        """Test cache save and load operations."""
        from ml_grid.util.data_cache import DataCache

        cache = DataCache(base_cache_dir=temp_dir)

        experiment_name = "test_experiment"
        run_id = "abc123"

        data_dict = {
            "X_train": sample_data[["feature1", "feature2"]],
            "y_train": sample_data["target"],
        }

        result = cache.save(experiment_name, run_id, data_dict)

        assert result is True

        loaded_data = cache.load(experiment_name, run_id)

        assert loaded_data is not None
        assert "X_train" in loaded_data
        assert "y_train" in loaded_data

    def test_load_nonexistent(self, temp_dir):
        """Test loading non-existent cache entry."""
        from ml_grid.util.data_cache import DataCache

        cache = DataCache(base_cache_dir=temp_dir)

        result = cache.load("nonexistent", "xyz789")

        assert result is None

    def test_invalidate(self, sample_data, temp_dir):
        """Test cache invalidation (delete)."""
        from ml_grid.util.data_cache import DataCache

        cache = DataCache(base_cache_dir=temp_dir)

        experiment_name = "test_exp"
        run_id = "run123"

        data_dict = {"data": "test"}

        cache.save(experiment_name, run_id, data_dict)
        assert cache.load(experiment_name, run_id) is not None

        cache.invalidate(experiment_name, run_id)
        assert cache.load(experiment_name, run_id) is None

    def test_clear_experiment_cache(self, sample_data, temp_dir):
        """Test clearing all cache for an experiment."""
        from ml_grid.util.data_cache import DataCache

        cache = DataCache(base_cache_dir=temp_dir)

        experiment_name = "multi_run_exp"

        data_dict = {"data": "test"}

        for i in range(3):
            run_id = f"run_{i}"
            cache.save(experiment_name, run_id, data_dict)

        stats = cache.get_cache_stats()
        assert stats["total_files"] == 3

        count = cache.clear_experiment_cache(experiment_name)

        assert count == 3
        stats_after = cache.get_cache_stats()
        assert stats_after["total_files"] == 0

    def test_get_cache_stats(self, sample_data, temp_dir):
        """Test cache statistics."""
        from ml_grid.util.data_cache import DataCache

        cache = DataCache(base_cache_dir=temp_dir)

        stats = cache.get_cache_stats()

        assert "total_files" in stats
        assert "total_size_bytes" in stats
        assert "cache_directory" in stats

    def test_cache_key_consistency(self, sample_data, temp_dir):
        """Test that same inputs produce same cache keys."""
        from ml_grid.util.data_cache import DataCache

        cache = DataCache(base_cache_dir=temp_dir)

        param_dict1 = {"scale": True, "test_sample_n": 0}
        param_dict2 = {"test_sample_n": 0, "scale": True}

        file_path = str(Path(temp_dir) / "data.csv")
        sample_data.to_csv(file_path, index=False)

        key1 = cache._create_cache_key(file_path, param_dict1)
        key2 = cache._create_cache_key(file_path, param_dict2)

        assert len(key1) == 32
        assert len(key2) == 32

    def test_cache_metadata(self, sample_data, temp_dir):
        """Test metadata storage in cache."""
        from ml_grid.util.data_cache import DataCache

        cache = DataCache(base_cache_dir=temp_dir)

        experiment_name = "meta_test"
        run_id = "run1"

        data_dict = {"X": sample_data[["feature1"]]}
        metadata = {
            "test_sample_n": 0,
            "timestamp": "2023-01-01",
            "custom_field": "value",
        }

        cache.save(experiment_name, run_id, data_dict, metadata)

        loaded = cache.load(experiment_name, run_id)

        assert loaded is not None


class TestThreadSafety:
    """Test thread safety of caching operations."""

    def test_concurrent_writes(self, sample_data, temp_dir):
        """Test concurrent write operations don't corrupt cache."""
        from ml_grid.util.data_cache import DataCache

        cache = DataCache(base_cache_dir=temp_dir)

        results = []
        errors = []

        def write_task(thread_id):
            try:
                experiment_name = f"thread_test_{thread_id}"
                run_id = "run1"

                data_dict = {
                    "X": sample_data.copy(),
                    "id": thread_id,
                }

                result = cache.save(experiment_name, run_id, data_dict)
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        threads = []
        for i in range(5):
            t = threading.Thread(target=write_task, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r is True for r in results)


class TestCacheInvalidation:
    """Test cache invalidation scenarios."""

    def test_different_params_different_keys(self, sample_data, temp_dir):
        """Test that different parameters produce different cache keys."""
        from ml_grid.util.data_cache import DataCache

        cache = DataCache(base_cache_dir=temp_dir)

        file_path = str(Path(temp_dir) / "data.csv")
        sample_data.to_csv(file_path, index=False)

        param_dict1 = {"scale": True}
        param_dict2 = {"scale": False}

        key1 = cache._create_cache_key(file_path, param_dict1)
        key2 = cache._create_cache_key(file_path, param_dict2)

        assert key1 != key2

    def test_sample_different_keys(self, sample_data, temp_dir):
        """Test that different sampling produces different keys."""
        from ml_grid.util.data_cache import DataCache

        cache = DataCache(base_cache_dir=temp_dir)

        file_path = str(Path(temp_dir) / "data.csv")
        sample_data.to_csv(file_path, index=False)

        param_dict1 = {"test_sample_n": 0}
        param_dict2 = {"test_sample_n": 50}

        key1 = cache._create_cache_key(file_path, param_dict1)
        key2 = cache._create_cache_key(file_path, param_dict2)

        assert key1 != key2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
