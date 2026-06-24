import sqlite3
import json
import logging
from typing import Any, Dict
import numpy as np

logger = logging.getLogger("ml_grid")


class DatabaseBackend:
    """Simple SQLite backend for storing ML experiment results."""

    def __init__(self, db_path: str = "ml_results.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        # Use a 30s timeout to handle concurrent writes during parallel runs
        return sqlite3.connect(self.db_path, timeout=30.0)

    def _init_db(self):
        """Initializes the results table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_timestamp TEXT,
                    date_time_stamp TEXT,
                    method_name TEXT,
                    outcome_variable TEXT,
                    auc REAL,
                    f1 REAL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    mcc REAL,
                    support REAL,
                    fit_time REAL,
                    score_time REAL,
                    run_time REAL,
                    nb_size INTEGER,
                    n_features INTEGER,
                    X_train_size INTEGER,
                    X_test_orig_size INTEGER,
                    X_test_size INTEGER,
                    n_fits INTEGER,
                    t_fits INTEGER,
                    param_space_index INTEGER,
                    parameters TEXT,
                    f_list TEXT,
                    resample TEXT,
                    scale TEXT,
                    percent_missing REAL,
                    corr REAL,
                    feature_selection_method TEXT,
                    use_embedding TEXT,
                    embedding_method TEXT,
                    embedding_dim INTEGER,
                    age TEXT,
                    sex TEXT,
                    bmi TEXT,
                    ethnicity TEXT,
                    bloods TEXT,
                    diagnostic_order TEXT,
                    drug_order TEXT,
                    annotation_n TEXT,
                    meta_sp_annotation_n TEXT,
                    meta_sp_annotation_mrc_n TEXT,
                    annotation_mrc_n TEXT,
                    core_02 TEXT,
                    bed TEXT,
                    vte_status TEXT,
                    hosp_site TEXT,
                    core_resus TEXT,
                    news TEXT,
                    failed TEXT,
                    timeout TEXT,
                    run_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _sanitize_value(self, value: Any) -> Any:
        """Converts numpy types and search space objects to native Python types for SQLite."""
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {
                self._sanitize_value(k): self._sanitize_value(v)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        # Handle non-serializable search space objects (skopt Integer, Real, Categorical)
        if value.__class__.__name__ in ["Integer", "Real", "Categorical"]:
            return str(value)
        return value

    def insert_result(self, result_data: Dict[str, Any]):
        """Inserts a single result record into the database.

        Args:
            result_data: Dictionary containing metric names and values.
        """
        # Define all expected columns in the database table
        columns = [
            "run_timestamp",
            "date_time_stamp",
            "method_name",
            "outcome_variable",
            "auc",
            "f1",
            "accuracy",
            "precision",
            "recall",
            "mcc",
            "support",
            "fit_time",
            "score_time",
            "run_time",
            "nb_size",
            "n_features",
            "X_train_size",
            "X_test_orig_size",
            "X_test_size",
            "n_fits",
            "t_fits",
            "param_space_index",
            "parameters",
            "f_list",
            "resample",
            "scale",
            "percent_missing",
            "corr",
            "feature_selection_method",
            "use_embedding",
            "embedding_method",
            "embedding_dim",
            "age",
            "sex",
            "bmi",
            "ethnicity",
            "bloods",
            "diagnostic_order",
            "drug_order",
            "annotation_n",
            "meta_sp_annotation_n",
            "meta_sp_annotation_mrc_n",
            "annotation_mrc_n",
            "core_02",
            "bed",
            "vte_status",
            "hosp_site",
            "core_resus",
            "news",
            "failed",
            "timeout",
            "run_path",
        ]

        # Map keys from result_data to columns, handling common aliases
        values = []
        for col in columns:
            val = result_data.get(col)

            # Key mapping for aliases used in project_score_save
            if val is None:
                if col == "run_timestamp":
                    val = result_data.get("date_time_stamp")
                elif col == "date_time_stamp":
                    val = result_data.get("run_timestamp")
                elif col == "parameters":
                    val = result_data.get("parameter_sample")
                elif col == "fit_time":
                    val = result_data.get("fit_time_m")
                elif col == "score_time":
                    val = result_data.get("score_time_m")
                elif col == "param_space_index":
                    val = result_data.get("i")
                elif col == "nb_size":
                    val = result_data.get("nb_size")
                elif col == "run_path":
                    val = "."

            # Apply recursive sanitization for numpy and search space types
            val = self._sanitize_value(val)

            # Explicitly handle boolean and flag columns to ensure they are stored as
            # "True"/"False" or "None" strings to match column type (TEXT).
            # This prevents sqlite3 from defaulting to 1/0 integers for bool types.
            if col in [
                "failed",
                "timeout",
                "age",
                "sex",
                "bmi",
                "ethnicity",
                "bloods",
                "diagnostic_order",
                "drug_order",
                "annotation_n",
                "meta_sp_annotation_n",
                "meta_sp_annotation_mrc_n",
                "annotation_mrc_n",
                "core_02",
                "bed",
                "vte_status",
                "hosp_site",
                "core_resus",
                "news",
                "resample",
                "scale",
                "use_embedding",
                "scale_features_before_embedding",
            ]:
                val = str(val)

            # Robust numeric casting for metric columns to prevent string pollution
            if col in [
                "auc",
                "f1",
                "accuracy",
                "precision",
                "recall",
                "mcc",
                "support",
                "fit_time",
                "score_time",
                "run_time",
                "percent_missing",
                "corr",
            ]:
                try:
                    val = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    val = 0.0
            elif col in [
                "nb_size",
                "n_features",
                "X_train_size",
                "X_test_orig_size",
                "X_test_size",
                "n_fits",
                "t_fits",
                "param_space_index",
                "embedding_dim",
            ]:
                try:
                    val = int(val) if val is not None else 0
                except (ValueError, TypeError):
                    val = 0

            # Ensure JSON columns have valid defaults if data is missing
            if col == "parameters" and val is None:
                val = {}
            if col == "f_list" and val is None:
                val = []

            if col in ["parameters", "f_list"] and isinstance(val, (dict, list)):
                val = json.dumps(val)

            values.append(val)

        placeholders = ", ".join(["?"] * len(columns))
        col_string = ", ".join(columns)
        query = f"INSERT INTO ml_results ({col_string}) VALUES ({placeholders})"

        try:
            with self._get_connection() as conn:
                conn.execute(query, values)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert result into database: {e}")
