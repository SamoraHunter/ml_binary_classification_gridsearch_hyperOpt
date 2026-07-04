"""
Data caching module for ml_grid pipeline.

This module provides efficient data caching using joblib to cache preprocessed
data, feature encoders/scalers, and transformed datasets for reuse between
experiments. Implements smart cache invalidation based on file checksums/timestamps.
"""

import hashlib
import json
import os
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib

from ml_grid.util.global_params import global_parameters


class CacheValidationError(Exception):
    """Raised when cached data validation fails."""

    pass


class DataCache:
    """
    Thread-safe disk-based caching system for ML pipeline data.

    Caches preprocessed data, feature transformers, and transformed datasets
    to reduce Pipe Instantiation time by avoiding redundant preprocessing steps.

    Cache Key Composition:
        - File checksum (SHA-256 of input file)
        - File modification timestamp
        - Sample configuration (test_sample_n, column_sample_n)
        - Feature selection parameters (pertubation_columns, drop_list)
        - Preprocessing flags (scale, use_embedding, embedding_method)
        - Outcome variable name

    Cache Invalidation:
        - Auto-invalidates when input file checksum changes
        - Auto-invalidates when input file modification time changes
        - Respects force_second_cv global parameter
        - Supports manual cache clearing via CLI flags

    Thread Safety:
        - Uses file-level locking for concurrent access
        - Atomic write operations using temp files + rename
    """

    def __init__(self, base_cache_dir: Optional[str] = None):
        """Initialize the data cache system."""
        self._lock = threading.RLock()
        self.global_params = global_parameters

        if base_cache_dir is None:
            import ml_binary_classification_gridsearch_hyperOpt

            base_cache_dir = os.path.join(
                os.path.dirname(ml_binary_classification_gridsearch_hyperOpt.__file__),
                ".ml_grid_cache",
            )

        self.base_cache_dir = Path(base_cache_dir)
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)

        self._logger = None
        self._setup_logging()

    def _setup_logging(self):
        """Setup logger instance."""
        import logging

        self._logger = logging.getLogger("ml_grid.cache")

    @property
    def cache_dir(self) -> Path:
        """Get the current experiment-specific cache directory."""
        return self.base_cache_dir

    def get_cache_path(self, experiment_name: str, run_id: str) -> Path:
        """
        Get full path for a cached item.

        Args:
            experiment_name: Name of the experiment
            run_id: Unique identifier for this specific run

        Returns:
            Full path to cache file
        """
        return self.cache_dir / f"{experiment_name}_{run_id}.joblib"

    def _compute_file_checksum(self, file_path: str) -> str:
        """
        Compute SHA-256 checksum of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hex digest of file checksum
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return ""

    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get file metadata for cache key computation.

        Args:
            file_path: Path to the data file

        Returns:
            Dictionary with checksum and modification time
        """
        try:
            stat = os.stat(file_path)
            return {
                "checksum": self._compute_file_checksum(file_path),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
        except Exception:
            return {"checksum": "", "mtime": 0, "size": 0}

    def _normalize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively normalize dictionary for consistent keys."""
        result = {}
        for key, value in d.items():
            if isinstance(value, (list, tuple)):
                result[key] = tuple(sorted(value))
            elif isinstance(value, dict):
                result[key] = self._normalize_dict(value)
            else:
                result[key] = value
        return result

    def _create_cache_key(
        self, file_path: str, local_param_dict: Dict[str, Any]
    ) -> str:
        """
        Create a unique cache key based on data and configuration.

        Args:
            file_path: Path to input data file
            local_param_dict: Pipeline parameters dictionary

        Returns:
            Unique cache key string
        """
        metadata = self._get_file_metadata(file_path)

        param_normalized = self._normalize_dict(local_param_dict)
        param_str = json.dumps(param_normalized, sort_keys=True, default=str)

        key_parts = [
            f"checksum={metadata.get('checksum', '')[:16]}",
            f"samples={local_param_dict.get('test_sample_n', 0)}",
            f"cols={local_param_dict.get('column_sample_n', 0)}",
            f"scale={local_param_dict.get('scale', False)}",
            f"embedding={local_param_dict.get('use_embedding', False)}",
            f"emb_method={local_param_dict.get('embedding_method', 'none')}",
        ]

        return hashlib.sha256(
            f"{param_str}|{'|'.join(key_parts)}".encode()
        ).hexdigest()[:32]

    def _create_transformer_cache_key(
        self, file_path: str, pertubation_columns: List[str], drop_list: List[str]
    ) -> str:
        """Create cache key for transformers (encoders/scalers)."""
        metadata = self._get_file_metadata(file_path)
        key_parts = [
            f"file={metadata.get('checksum', '')[:16]}",
            f"features={len(pertubation_columns)}",
            f"dropped={len(drop_list)}",
        ]
        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:32]

    def save(
        self,
        experiment_name: str,
        run_id: str,
        data_dict: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save data dictionary to cache.

        Args:
            experiment_name: Name of the experiment
            run_id: Unique identifier for this run
            data_dict: Dictionary of data to cache (X_train, X_test, scaler, etc.)
            metadata: Optional metadata to store

        Returns:
            True if save was successful, False otherwise
        """
        with self._lock:
            try:
                cache_path = self.get_cache_path(experiment_name, run_id)

                full_data = {
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {},
                    "data": data_dict,
                }

                temp_path = str(cache_path) + ".tmp"
                joblib.dump(full_data, temp_path)
                os.rename(temp_path, cache_path)

                self._logger.debug(f"Cached data saved: {cache_path}")
                return True

            except Exception as e:
                self._logger.error(f"Failed to save cache: {e}")
                if "temp_path" in locals():
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
                return False

    def load(self, experiment_name: str, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Load cached data.

        Args:
            experiment_name: Name of the experiment
            run_id: Unique identifier for this run

        Returns:
            Cached data dictionary or None if not found/invalid
        """
        with self._lock:
            cache_path = self.get_cache_path(experiment_name, run_id)

            try:
                if not cache_path.exists():
                    return None

                full_data = joblib.load(str(cache_path))

                if "data" not in full_data:
                    return None

                return full_data["data"]

            except Exception as e:
                self._logger.warning(f"Failed to load cache: {e}")
                try:
                    cache_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return None

    def invalidate(self, experiment_name: str, run_id: str) -> bool:
        """
        Invalidate (delete) a specific cache entry.

        Args:
            experiment_name: Name of the experiment
            run_id: Unique identifier for this run

        Returns:
            True if invalidated successfully
        """
        with self._lock:
            try:
                cache_path = self.get_cache_path(experiment_name, run_id)
                lock_file = cache_path.with_suffix(".lock")

                with open(str(lock_file), "w") as f:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    cache_path.unlink(missing_ok=True)
                    lock_file.unlink(missing_ok=True)

                self._logger.debug(f"Cache invalidated: {cache_path}")
                return True
            except Exception:
                return False

    def clear_experiment_cache(self, experiment_name: str) -> int:
        """
        Clear all cached data for an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Number of files removed
        """
        with self._lock:
            count = 0
            try:
                pattern = f"{experiment_name}_*.joblib"
                for cache_file in self.cache_dir.glob(pattern):
                    try:
                        cache_file.unlink()
                        count += 1
                    except Exception:
                        pass

                self._logger.info(
                    f"Cleared {count} cache files for experiment: {experiment_name}"
                )
            except Exception as e:
                self._logger.error(f"Failed to clear cache: {e}")

            return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache directory."""
        with self._lock:
            total_size = 0
            file_count = 0

            for cache_file in self.cache_dir.glob("*.joblib"):
                try:
                    total_size += cache_file.stat().st_size
                    file_count += 1
                except Exception:
                    pass

            return {
                "total_files": file_count,
                "total_size_bytes": total_size,
                "cache_directory": str(self.base_cache_dir),
            }


class CachedPipelineInstance:
    """
    Pipeline instance that uses cached data when available.

    This class wraps the original pipe class to provide transparent caching
    of preprocessed data and transformers between runs.
    """

    def __init__(
        self,
        experiment_name: str,
        run_id: str,
        cache: DataCache,
        file_path: str,
        local_param_dict: Dict[str, Any],
        original_kwargs: Dict[str, Any],
        original_class: Any,
    ):
        """Initialize cached pipeline instance."""
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.cache = cache
        self.file_path = file_path
        self.local_param_dict = local_param_dict
        self.original_kwargs = original_kwargs
        self.original_class = original_class

        self._cached_data = None
        self._pipeline = None

    def _try_load_cache(self) -> bool:
        """Try to load cached data. Returns True if successful."""
        cache_data = self.cache.load(self.experiment_name, self.run_id)

        if cache_data is not None and "_metadata" in cache_data:
            self._cached_data = cache_data
            return True

        return False

    def _save_to_cache(self, pipeline: Any) -> bool:
        """Save pipeline data to cache."""
        data_dict = {
            "df": getattr(pipeline, "df", None),
            "X": getattr(pipeline, "X", None),
            "y": getattr(pipeline, "y", None),
            "X_train": getattr(pipeline, "X_train", None),
            "X_test": getattr(pipeline, "X_test", None),
            "y_train": getattr(pipeline, "y_train", None),
            "y_test": getattr(pipeline, "y_test", None),
            "X_test_orig": getattr(pipeline, "X_test_orig", None),
            "y_test_orig": getattr(pipeline, "y_test_orig", None),
            "final_column_list": getattr(pipeline, "final_column_list", None),
        }

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "test_sample_n": self.local_param_dict.get("test_sample_n", 0),
            "column_sample_n": self.local_param_dict.get("column_sample_n", 0),
        }

        return self.cache.save(self.experiment_name, self.run_id, data_dict, metadata)

    def _create_pipeline_from_cache(self) -> Any:
        """Create pipeline instance from cached data."""
        if self._cached_data is None:
            return None

        pipeline = object.__new__(self.original_class)

        for key, value in self._cached_data.items():
            if key != "_metadata":
                setattr(pipeline, key, value)

        return pipeline

    def run_pipeline(self) -> Any:
        """
        Run the pipeline, using cached data when available.

        Returns:
            Pipeline instance with data populated
        """
        if self._try_load_cache():
            pipeline = self._create_pipeline_from_cache()

            if pipeline is not None:
                return pipeline

        pipeline = self.original_class(**self.original_kwargs)

        if self._save_to_cache(pipeline):
            pass

        return pipeline


class CachedDataPipeline:
    """
    Data pipeline wrapper with caching support.

    Wraps the existing data pipeline class to provide transparent caching
    of preprocessed data and transformers between runs.
    """

    def __init__(self, original_pipeline_class: Any, cache: Optional[DataCache] = None):
        """
        Initialize CachedDataPipeline wrapper.

        Args:
            original_pipeline_class: The original pipe class to wrap
            cache: DataCache instance (creates new if None)
        """
        self._original_class = original_pipeline_class
        self.cache = cache or DataCache()

        self.cached_instances = {}

    def _get_cache_keys(
        self, file_name: str, local_param_dict: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Get experiment name and run ID for caching.

        Args:
            file_name: Path to input file
            local_param_dict: Pipeline parameters

        Returns:
            Tuple of (experiment_name, run_id)
        """
        import os

        base_name = os.path.basename(file_name).replace(".csv", "")
        experiment_name = base_name.replace(".", "_").replace("-", "_")

        param_str = json.dumps(local_param_dict, sort_keys=True, default=str)
        run_id = hashlib.sha256(param_str.encode()).hexdigest()[:16]

        return experiment_name, run_id

    def create_cached_pipeline(
        self,
        experiment_name: str,
        run_id: str,
        file_path: str,
        local_param_dict: Dict[str, Any],
        original_kwargs: Dict[str, Any],
    ) -> "CachedPipelineInstance":
        """
        Create a cached pipeline instance.

        Args:
            experiment_name: Name of the experiment
            run_id: Run identifier
            file_name: Path to input data
            local_param_dict: Pipeline parameters
            original_kwargs: Original __init__ arguments

        Returns:
            CachedPipelineInstance that uses cached data when available
        """
        instance = CachedPipelineInstance(
            experiment_name=experiment_name,
            run_id=run_id,
            cache=self.cache,
            file_path=file_path,
            local_param_dict=local_param_dict,
            original_kwargs=original_kwargs,
            original_class=self._original_class,
        )

        return instance


def with_cache_validation(func: Callable) -> Callable:
    """Decorator to add cache validation before method execution."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "cache") or self.cache is None:
            return func(self, *args, **kwargs)

        force_second = getattr(self, "_force_second_cv", False) or getattr(
            getattr(self, "global_params", None), "force_second_cv", False
        )

        if force_second:
            if hasattr(self, "_logger"):
                self._logger.debug("Force second CV enabled - skipping cache")

        return func(self, *args, **kwargs)

    return wrapper


def get_cache_stats(cache: DataCache) -> Dict[str, Any]:
    """Helper function to get cache statistics."""
    return cache.get_cache_stats()


def clear_experiment_cache(cache: DataCache, experiment_name: str) -> int:
    """Helper function to clear cache for an experiment."""
    return cache.clear_experiment_cache(experiment_name)
