import unittest
import sqlite3
import os
import tempfile
import json
import numpy as np
from pathlib import Path
from ml_grid.util.db_backend import DatabaseBackend


class TestDatabaseBackend(unittest.TestCase):

    def setUp(self):
        # Create a temporary database file for each test
        self.temp_db_file = Path(
            tempfile.NamedTemporaryFile(delete=False, suffix=".db").name
        )
        self.db_backend = DatabaseBackend(db_path=str(self.temp_db_file))

    def tearDown(self):
        # Clean up the temporary database file after each test
        if self.temp_db_file.exists():
            os.remove(self.temp_db_file)

    def test_init_db_creates_table(self):
        # Verify that the table exists after initialization
        with sqlite3.connect(self.temp_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ml_results';"
            )
            self.assertIsNotNone(
                cursor.fetchone(), "Table 'ml_results' should be created"
            )

            # Verify schema (check a few columns)
            cursor.execute("PRAGMA table_info(ml_results);")
            columns = [col[1] for col in cursor.fetchall()]  # col[1] is the column name
            self.assertIn("id", columns)
            self.assertIn("run_timestamp", columns)
            self.assertIn("auc", columns)
            self.assertIn("parameters", columns)
            self.assertIn("nb_size", columns)
            self.assertIn("created_at", columns)

    def test_sanitize_value(self):
        # Test numpy types
        self.assertAlmostEqual(
            self.db_backend._sanitize_value(np.float32(1.23)), 1.23, places=6
        )
        self.assertEqual(self.db_backend._sanitize_value(np.float64(4.56)), 4.56)
        self.assertEqual(self.db_backend._sanitize_value(np.int32(7)), 7)
        self.assertEqual(self.db_backend._sanitize_value(np.int64(8)), 8)
        self.assertEqual(
            self.db_backend._sanitize_value(np.array([1, 2, 3])), [1, 2, 3]
        )
        self.assertEqual(
            self.db_backend._sanitize_value(np.array([[1, 2], [3, 4]])),
            [[1, 2], [3, 4]],
        )

        # Test native python types (should remain unchanged)
        self.assertEqual(self.db_backend._sanitize_value(1.0), 1.0)
        self.assertEqual(self.db_backend._sanitize_value(1), 1)
        self.assertEqual(self.db_backend._sanitize_value("test"), "test")
        self.assertEqual(self.db_backend._sanitize_value([1, 2]), [1, 2])
        self.assertEqual(self.db_backend._sanitize_value({"a": 1}), {"a": 1})
        self.assertIsNone(self.db_backend._sanitize_value(None))

    def test_insert_result_full_data(self):
        test_data = {
            "run_timestamp": "2023-01-01 12:00:00",
            "method_name": "TestModel",
            "outcome_variable": "target_outcome",
            "auc": 0.85,
            "f1": 0.78,
            "accuracy": 0.80,
            "precision": 0.75,
            "recall": 0.82,
            "mcc": 0.65,
            "fit_time": 120.5,
            "nb_size": 100,
            "parameters": {"param1": "value1", "param2": 123},
            "failed": False,
            "run_path": "/path/to/run",
        }
        self.db_backend.insert_result(test_data)

        with sqlite3.connect(self.temp_db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ml_results WHERE method_name='TestModel';")
            result = cursor.fetchone()
            self.assertIsNotNone(result)

            self.assertEqual(result["run_timestamp"], test_data["run_timestamp"])
            self.assertEqual(result["method_name"], test_data["method_name"])
            self.assertEqual(result["auc"], test_data["auc"])
            self.assertEqual(result["fit_time"], test_data["fit_time"])
            self.assertEqual(result["nb_size"], test_data["nb_size"])
            self.assertEqual(json.loads(result["parameters"]), test_data["parameters"])
            self.assertEqual(result["failed"], str(test_data["failed"]))
            self.assertEqual(result["run_path"], test_data["run_path"])

    def test_insert_result_minimal_data(self):
        minimal_data = {
            "method_name": "MinimalModel",
            "outcome_variable": "min_target",
            "auc": 0.6,
        }
        self.db_backend.insert_result(minimal_data)

        with sqlite3.connect(self.temp_db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ml_results WHERE method_name='MinimalModel';")
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result["method_name"], minimal_data["method_name"])
            self.assertEqual(result["auc"], minimal_data["auc"])
            # Check default values
            self.assertEqual(result["f1"], 0.0)
            self.assertEqual(result["nb_size"], 0)
            self.assertEqual(json.loads(result["parameters"]), {})
            self.assertEqual(
                result["failed"], "None"
            )  # result_data.get returns None if missing

    def test_insert_result_with_numpy_types(self):
        np_data = {
            "run_timestamp": "2023-01-02 10:00:00",
            "method_name": "NumpyModel",
            "outcome_variable": "np_target",
            "auc": np.float64(0.91),
            "f1": np.float32(0.88),
            "nb_size": np.int64(150),
            "parameters": {
                "array_param": np.array([0.1, 0.2, 0.3]),
                "int64_param": np.int64(99),
            },  # Added int64_param
            "failed": True,
        }
        self.db_backend.insert_result(np_data)

        with sqlite3.connect(self.temp_db_file) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ml_results WHERE method_name='NumpyModel';")
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result["auc"], float(np_data["auc"]))
            self.assertEqual(result["f1"], float(np_data["f1"]))
            self.assertEqual(result["nb_size"], int(np_data["nb_size"]))
            deserialized_params = json.loads(result["parameters"])
            self.assertEqual(
                deserialized_params["array_param"],
                np_data["parameters"]["array_param"].tolist(),
            )
            self.assertEqual(
                deserialized_params["int64_param"],
                int(np_data["parameters"]["int64_param"]),
            )  # Assert on converted type
            self.assertEqual(result["failed"], "True")

    def test_insert_result_parameters_as_list(self):
        list_params_data = {
            "method_name": "ListParamModel",
            "outcome_variable": "list_target",
            "auc": 0.7,
            "parameters": ["param_a", "param_b"],
        }
        self.db_backend.insert_result(list_params_data)

        with sqlite3.connect(self.temp_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT parameters FROM ml_results WHERE method_name='ListParamModel';"
            )
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(json.loads(result[0]), list_params_data["parameters"])

    def test_insert_result_parameters_as_string(self):
        string_params_data = {
            "method_name": "StringParamModel",
            "outcome_variable": "string_target",
            "auc": 0.75,
            "parameters": "some_string_representation",
        }
        self.db_backend.insert_result(string_params_data)

        with sqlite3.connect(self.temp_db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT parameters FROM ml_results WHERE method_name='StringParamModel';"
            )
            result = cursor.fetchone()
            self.assertIsNotNone(result)
            # Should not be JSON-decoded if it's already a string
            self.assertEqual(result[0], string_params_data["parameters"])

    def test_insert_result_error_handling(self):
        # The current _sanitize_value and .get() with defaults make it hard to force a DB error
        # with malformed input directly. It will just insert defaults or sanitized values.
        # So, this test primarily ensures the method doesn't crash.
        invalid_data = {
            "method_name": "ErrorModel",
            "outcome_variable": "error_target",
            "auc": "not_a_number",  # This will be sanitized to 0.0 by _sanitize_value(result_data.get("auc", 0.0))
            "fit_time": "also_not_a_number",  # This will be sanitized to 0.0
            "nb_size": "not_an_int",  # This will be sanitized to 0
        }
        try:
            self.db_backend.insert_result(invalid_data)
            with sqlite3.connect(self.temp_db_file) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT auc, fit_time, nb_size FROM ml_results WHERE method_name='ErrorModel';"
                )
                result = cursor.fetchone()
                self.assertIsNotNone(result)
                self.assertEqual(result["auc"], 0.0)
                self.assertEqual(result["fit_time"], 0.0)
                self.assertEqual(result["nb_size"], 0)
        except Exception as e:
            self.fail(f"insert_result raised an unexpected exception: {e}")

    def test_concurrent_access_timeout(self):
        # This test is more conceptual for a unit test, as true concurrency
        # requires multiple processes. However, we can verify the timeout is set.
        with sqlite3.connect(self.temp_db_file, timeout=0.1) as conn1:
            # Lock the database by starting a transaction but not committing
            conn1.execute("BEGIN IMMEDIATE;")
            # Try to perform an operation in another connection with a short timeout
            with self.assertRaises(sqlite3.OperationalError) as cm:
                with sqlite3.connect(self.temp_db_file, timeout=0.05) as conn2:
                    # This operation will fail because the database is locked by conn1
                    conn2.execute(
                        "INSERT INTO ml_results (method_name) VALUES ('LockedAttempt');"
                    )
            self.assertIn("database is locked", str(cm.exception))
            # The actual DatabaseBackend uses a 30s timeout, which is long enough
            # for most operations to complete without contention in a real scenario.
            # This test confirms that SQLite's timeout mechanism works.
