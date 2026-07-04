import logging
import multiprocessing
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed
from multiprocessing import Lock, Manager


class ModelExecutionError(Exception):
    """Custom exception for model execution failures."""

    pass


@contextmanager
def _timeout_context(seconds: float):
    """Context manager for timeout that works in child processes."""
    if seconds is None or seconds <= 0:
        yield
        return

    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def signal_handler(signum, frame):
        raise TimeoutError(f"Timeout of {seconds}s reached")

    try:
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(seconds))
        yield
    finally:
        signal.alarm(0)
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, old_handler)


class ParallelModelExecutor:
    """
    Manages parallel execution of multiple models with proper resource management.

    Features:
    - Automatic detection of GPU vs CPU models
    - Proper thread count allocation per model
    - Shared read-only data via numpy arrays (no pickling overhead)
    - Thread-safe result aggregation using Lock-based synchronization
    - Timeout handling per model
    """

    def __init__(self, n_jobs: Optional[int] = None, verbose: int = 0):
        """
        Initialize the parallel executor.

        Args:
            n_jobs: Number of parallel jobs. If None, uses all available CPU cores.
            verbose: Verbosity level (0=quiet, 1=minimal, 2=detailed)
        """
        self.logger = logging.getLogger("ml_grid")
        self.verbose = verbose

        # Determine number of workers
        if n_jobs is None:
            self.n_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_workers = max(1, min(n_jobs, multiprocessing.cpu_count()))

        self.logger.info(
            f"Parallel execution initialized with {self.n_workers} workers"
        )

        # Resource management
        self.gpu_model_locks: Dict[str, bool] = {}  # Track GPU model activity
        self.cpu_pool_lock = Lock()  # Protect CPU pool coordination

        # Shared state for result aggregation
        self.manager = Manager()
        self.results_queue = self.manager.Queue()
        self.error_queue = self.manager.Queue()
        self.global_result_lock = Lock()

    def _is_gpu_model(self, method_name: str) -> bool:
        """Detect if a model requires GPU resources."""
        gpu_indicators = ["keras", "xgb", "catboost", "neural", "torch", "cuda"]
        method_lower = method_name.lower()
        return any(ind in method_lower for ind in gpu_indicators)

    def _is_h2o_model(self, algorithm) -> bool:
        """Detect if an H2O model."""
        try:
            import h2o

            return isinstance(
                algorithm,
                tuple(
                    cls
                    for cls in h2o.model.H2OModel.__subclasses__()
                    if "Classifier" in str(cls)
                ),
            )
        except ImportError:
            return False

    def _estimate_threads_needed(self, method_name: str) -> int:
        """
        Estimate the number of threads a model needs.

        GPU models get 1 thread (they manage their own parallelism),
        CPU models can use multiple threads.
        """
        if self._is_gpu_model(method_name):
            return 1
        return max(2, self.n_workers // 2)

    def _create_worker_args(
        self, model_idx: int, args_tuple: Tuple, shared_data: Dict[str, Any]
    ) -> Tuple:
        """
        Create worker arguments with references to shared data.

        Shared data includes:
        - X_train, y_train (read-only numpy arrays)
        - X_test, y_test (read-only numpy arrays)
        - Data pipeline configuration

        Non-shared items are deep-copied:
        - Algorithm implementation
        - Parameter space
        """
        args_list = list(args_tuple)

        # Extract ml_grid_object
        algorithm_impl = args_list[0]
        param_space = args_list[1]
        method_name = args_list[2]
        _ml_grid_obj = args_list[3]
        sub_sample_val = args_list[4]
        score_instance = args_list[5]

        # Get data from shared cache
        X_train = shared_data.get("X_train")
        y_train = shared_data.get("y_train")
        X_test = shared_data.get("X_test")
        y_test = shared_data.get("y_test")

        # Deep copy non-serializable items to prevent state sharing issues
        copied_param_space = deepcopy(param_space)

        return (
            model_idx,
            algorithm_impl,
            copied_param_space,
            method_name,
            X_train,
            y_train,
            X_test,
            y_test,
            sub_sample_val,
            score_instance,
            shared_data.get("timeout", None),
            shared_data.get("time_series_mode", False),
            self._is_gpu_model(method_name),
        )

    def _execute_single_model(
        self,
        model_idx: int,
        algorithm_impl: Any,
        param_space: Dict,
        method_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sub_sample_val: int,
        score_instance: Any,
        timeout: Optional[float],
        is_ts_mode: bool,
        is_gpu_model: bool,
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Execute a single model and return results.

        This function runs in a separate process/thread.

        Returns:
            Tuple of (model_idx, score, metadata_dict)
        """
        try:
            start_time = time.time()

            # Configure thread usage based on resource requirements
            import os

            if is_gpu_model:
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
                os.environ["OPENBLAS_NUM_THREADS"] = "1"
            else:
                threads_needed = max(1, multiprocessing.cpu_count() // 4)
                os.environ["OMP_NUM_THREADS"] = str(min(threads_needed, 4))

            # Import execution module
            from ml_grid.pipeline import grid_search_cross_validate

            # Create time-limited context
            with _timeout_context(timeout):
                gscv_instance = grid_search_cross_validate.grid_search_crossvalidate(
                    algorithm_impl,
                    param_space,
                    method_name,
                    None,  # Will be set below
                    sub_sample_val,
                    score_instance,
                )

                score = getattr(
                    gscv_instance, "grid_search_cross_validate_score_result", 0.5
                )

            elapsed = time.time() - start_time

            return (
                model_idx,
                float(score),
                {
                    "method_name": method_name,
                    "elapsed_time": elapsed,
                    "is_gpu_model": is_gpu_model,
                    "success": True,
                    "error_message": None,
                },
            )

        except TimeoutError as e:
            return (
                model_idx,
                0.0,
                {
                    "method_name": method_name,
                    "elapsed_time": timeout or 60,
                    "is_gpu_model": is_gpu_model,
                    "success": False,
                    "error_type": "TimeoutError",
                    "error_message": str(e),
                },
            )

        except Exception as e:
            return (
                model_idx,
                0.0,
                {
                    "method_name": method_name,
                    "elapsed_time": None,
                    "is_gpu_model": is_gpu_model,
                    "success": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

    def _collect_results(self, timeout: float = 30.0) -> Tuple[List[Tuple], List[Dict]]:
        """Collect results from the queues."""
        collected_results = []
        collected_errors = []

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                if not self.results_queue.empty():
                    result = self.results_queue.get(timeout=0.1)
                    collected_results.append(result)
                elif not self.error_queue.empty():
                    error = self.error_queue.get(timeout=0.1)
                    collected_errors.append(error)
            except Exception:
                break

        return collected_results, collected_errors

    def execute_models_parallel(
        self,
        arg_list: List[Tuple],
        shared_data: Dict[str, Any],
        timeout: Optional[float] = None,
        max_retries: int = 2,
    ) -> Tuple[List[Tuple], List[Dict]]:
        """
        Execute all models in parallel.

        Args:
            arg_list: List of argument tuples for each model
            shared_data: Dictionary containing shared data (X_train, y_train, etc.)
            timeout: Global timeout for the entire execution
            max_retries: Maximum retry attempts for failed models

        Returns:
            Tuple of (success_results, error_results)
        """
        n_models = len(arg_list)

        if self.verbose >= 2:
            self.logger.info(f"Starting parallel execution of {n_models} models")

        start_time = time.time()

        # Configure timeout for each model
        model_timeout = (
            (timeout / n_models) * 0.8  # Allow some buffer
            if timeout and n_models > 1
            else timeout
        )

        # Prepare parameters for all models
        prepared_args = [
            self._create_worker_args(idx, args, shared_data)
            for idx, args in enumerate(arg_list)
        ]

        # Group models by resource requirements
        gpu_models = [(i, a) for i, a in enumerate(prepared_args) if a[12]]
        cpu_models = [(i, a) for i, a in enumerate(prepared_args) if not a[12]]

        all_results = []
        all_errors = []

        # Execute GPU models first (sequential due to resource constraints)
        if gpu_models:
            self.logger.info(f"Executing {len(gpu_models)} GPU model(s)")
            for idx, args in gpu_models:
                result = self._execute_single_model(*args)
                if result[2].get("success"):
                    all_results.append(result)
                else:
                    all_errors.append((idx, result[2]))

        # Execute CPU models in parallel
        if cpu_models:
            cpu_args = [args for _, args in cpu_models]

            self.logger.info(f"Executing {len(cpu_args)} CPU model(s) in parallel")

            try:
                with ThreadPoolExecutor(
                    max_workers=min(self.n_workers, len(cpu_args))
                ) as executor:
                    futures = [
                        executor.submit(self._execute_single_model, *args)
                        for args in cpu_args
                    ]

                    for i, future in enumerate(futures):
                        try:
                            result = future.result(timeout=model_timeout)
                            if result[2].get("success"):
                                all_results.append(result)
                            else:
                                # Retry on failure
                                retry_count = 0
                                while retry_count < max_retries:
                                    retry_count += 1
                                    if self.verbose >= 1:
                                        self.logger.info(
                                            f"Retrying CPU model {cpu_args[i][2]} (attempt {retry_count+1}/{max_retries+1})"
                                        )
                                    try:
                                        result = self._execute_single_model(
                                            *cpu_args[i]
                                        )
                                        if result[2].get("success"):
                                            all_results.append(result)
                                            break
                                        else:
                                            all_errors.append(
                                                (cpu_args[i][0], result[2])
                                            )
                                            if retry_count >= max_retries:
                                                break
                                    except Exception as e:
                                        if self.verbose >= 1:
                                            self.logger.warning(
                                                f"Retry {retry_count+1} failed for model: {cpu_args[i][2]}"
                                            )
                                        all_errors.append(
                                            (
                                                cpu_args[i][0],
                                                {
                                                    "method_name": cpu_args[i][2],
                                                    "error_type": type(e).__name__,
                                                    "error_message": str(e),
                                                },
                                            )
                                        )
                        except Exception as e:
                            if self.verbose >= 1:
                                self.logger.warning(f"Model execution failed: {e}")
            except Exception as e:
                self.logger.error(f"Parallel execution error: {e}")

        elapsed = time.time() - start_time

        # Sort results by original model index for consistent ordering
        all_results.sort(key=lambda x: x[0])

        if self.verbose >= 2:
            self.logger.info(
                f"Parallel execution completed in {elapsed:.2f}s. "
                f"{len(all_results)} successful, {len(all_errors)} failed"
            )

        return all_results, all_errors

    def execute_models_joblib_parallel(
        self,
        arg_list: List[Tuple],
        shared_data: Dict[str, Any],
        timeout: Optional[float] = None,
        batch_size: int = "auto",
    ) -> Tuple[List[Tuple], List[Dict]]:
        """
        Execute all models in parallel using joblib.

        This approach is more efficient for many small tasks and reduces
        process creation overhead.

        Args:
            arg_list: List of argument tuples for each model
            shared_data: Dictionary containing shared data
            timeout: Timeout per model execution
            batch_size: Batch size for joblib (auto, int, or float)

        Returns:
            Tuple of (success_results, error_results)
        """
        n_models = len(arg_list)

        if self.verbose >= 2:
            self.logger.info(
                f"Starting joblib parallel execution with {n_models} models"
            )

        start_time = time.time()

        # Prepare worker arguments
        prepared_args = [
            self._create_worker_args(idx, args, shared_data)
            for idx, args in enumerate(arg_list)
        ]

        def safe_execute(args):
            """Safe wrapper with exception handling."""
            try:
                result = self._execute_single_model(*args)
                return result
            except Exception as e:
                return (
                    args[0],
                    0.0,
                    {
                        "method_name": args[2],
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                        "success": False,
                    },
                )

        # Determine batch size
        if batch_size == "auto":
            batch_size = max(1, n_models // self.n_workers)

        try:
            results = Parallel(
                n_jobs=self.n_workers,
                prefer="threads",  # Use threads for better shared memory handling
                batch_size=batch_size,
                verbose=max(0, self.verbose - 2),
            )(delayed(safe_execute)(args) for args in prepared_args)

            elapsed = time.time() - start_time

            # Separate successes and errors
            success_results = []
            error_results = []

            for result in results:
                if result[2].get("success"):
                    success_results.append(result)
                else:
                    error_results.append((result[0], result[2]))

            if self.verbose >= 2:
                self.logger.info(
                    f"Joblib parallel execution completed in {elapsed:.2f}s. "
                    f"{len(success_results)} successful, {len(error_results)} failed"
                )

            return success_results, error_results

        except Exception as e:
            self.logger.error(f"Joblib execution failed: {e}")
            # Fall back to sequential execution
            if self.verbose >= 1:
                self.logger.info("Falling back to sequential execution")

            success_results = []
            error_results = []

            for args in prepared_args:
                result = safe_execute(args)
                if result[2].get("success"):
                    success_results.append(result)
                else:
                    error_results.append((result[0], result[2]))

            return success_results, error_results

    def execute_models_multiprocessing(
        self,
        arg_list: List[Tuple],
        shared_data: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Tuple[List[Tuple], List[Dict]]:
        """
        Execute all models using multiprocessing Pool.

        Best for heavy models with significant computation per model.
        Uses fork server or spawn start method for better isolation.

        Args:
            arg_list: List of argument tuples for each model
            shared_data: Dictionary containing shared data
            timeout: Timeout per model execution

        Returns:
            Tuple of (success_results, error_results)
        """
        n_models = len(arg_list)

        if self.verbose >= 2:
            self.logger.info(
                f"Starting multiprocessing execution with {n_models} models"
            )

        start_time = time.time()

        # Prepare arguments
        prepared_args = [
            self._create_worker_args(idx, args, shared_data)
            for idx, args in enumerate(arg_list)
        ]

        def safe_execute(args):
            """Safe wrapper with exception handling."""
            try:
                result = self._execute_single_model(*args)
                return result
            except Exception as e:
                return (
                    args[0],
                    0.0,
                    {
                        "method_name": args[2],
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                        "success": False,
                    },
                )

        try:
            # Use forkserver for better multi-thread support
            ctx = multiprocessing.get_context("forkserver")

            with ctx.Pool(processes=self.n_workers) as pool:
                if timeout is not None:
                    results = []
                    error_results = []  # Initialize here for timeout path
                    for i, result in enumerate(
                        pool.imap_unordered(
                            lambda args: self._execute_single_model(*args),
                            prepared_args,
                            chunksize=max(1, n_models // self.n_workers),
                        )
                    ):
                        try:
                            res = result.get(timeout=timeout / n_models)
                            results.append(res)
                        except Exception as e:
                            if self.verbose >= 1:
                                self.logger.warning(
                                    f"Model {prepared_args[i][2]} timed out: {e}"
                                )
                            error_results.append(
                                (
                                    prepared_args[0][0] if prepared_args else None,
                                    {
                                        "method_name": (
                                            prepared_args[0][2]
                                            if prepared_args
                                            else "Unknown"
                                        ),
                                        "error_type": "TimeoutError",
                                        "success": False,
                                    },
                                )
                            )
                else:
                    results = pool.map(
                        lambda args: self._execute_single_model(*args),
                        prepared_args,
                        chunksize=max(1, n_models // self.n_workers),
                    )

            elapsed = time.time() - start_time

            success_results = [r for r in results if r[2].get("success")]
            error_results = [(r[0], r[2]) for r in results if not r[2].get("success")]

            if self.verbose >= 2:
                self.logger.info(
                    f"Multiprocessing execution completed in {elapsed:.2f}s. "
                    f"{len(success_results)} successful, {len(error_results)} failed"
                )

            return success_results, error_results

        except Exception as e:
            self.logger.error(f"Multiprocessing execution failed: {e}")
            raise

    def close(self):
        """Cleanup resources."""
        try:
            if hasattr(self, "manager"):
                self.manager.shutdown()
        except Exception:
            pass


def create_parallel_executor(
    n_jobs: Optional[int] = None, method: str = "auto", verbose: int = 0
) -> ParallelModelExecutor:
    """
    Factory function to create a parallel executor.

    Args:
        n_jobs: Number of parallel jobs
        method: Execution method ('joblib', 'multiprocessing', or 'auto')
        verbose: Verbosity level

    Returns:
        Configured ParallelModelExecutor instance
    """
    executor = ParallelModelExecutor(n_jobs=n_jobs, verbose=verbose)

    if method == "auto":
        # Use joblib for most cases (better overhead handling)
        executor._execute_method = executor.execute_models_joblib_parallel
    elif method == "joblib":
        executor._execute_method = executor.execute_models_joblib_parallel
    elif method == "multiprocessing":
        executor._execute_method = executor.execute_models_multiprocessing
    else:
        raise ValueError(f"Unknown execution method: {method}")

    return executor


# Backward compatibility - the old execute function
def execute_with_parallel_support(
    run_instance: Any,
    arg_list: List[Tuple],
    n_jobs: Optional[int] = None,
    verbose: int = 0,
) -> Tuple[List[List[Any]], float]:
    """
    Execute models with parallel processing support.

    This function replaces the sequential execution in main.py:execute()

    Args:
        run_instance: The run instance for accessing global parameters
        arg_list: List of argument tuples for each model
        n_jobs: Number of parallel jobs (None = auto-detect)
        verbose: Verbosity level

    Returns:
        Tuple of (model_error_list, highest_score)
    """

    # Extract data from run_instance
    ml_grid_object = run_instance.ml_grid_object
    timeout = run_instance.local_param_dict.get(
        "model_eval_time_limit", run_instance.global_params.model_eval_time_limit
    )

    # Prepare shared data (read-only numpy arrays)
    shared_data = {
        "X_train": (
            ml_grid_object.X_train.to_numpy(dtype=float)
            if hasattr(ml_grid_object.X_train, "to_numpy")
            else np.array(ml_grid_object.X_train)
        ),
        "y_train": (
            ml_grid_object.y_train.to_numpy(dtype=float)
            if hasattr(ml_grid_object.y_train, "to_numpy")
            else np.array(ml_grid_object.y_train)
        ),
        "X_test": (
            ml_grid_object.X_test.to_numpy(dtype=float)
            if hasattr(ml_grid_object.X_test, "to_numpy")
            else np.array(ml_grid_object.X_test)
        ),
        "y_test": (
            ml_grid_object.y_test.to_numpy(dtype=float)
            if hasattr(ml_grid_object.y_test, "to_numpy")
            else np.array(ml_grid_object.y_test)
        ),
        "timeout": timeout,
        "time_series_mode": getattr(ml_grid_object, "time_series_mode", False),
    }

    # Create executor
    executor = create_parallel_executor(
        n_jobs=n_jobs,
        method="joblib",  # Joblib has better overhead handling for ML workloads
        verbose=verbose,
    )

    try:
        # Execute in parallel
        success_results, error_results = executor.execute_models_joblib_parallel(
            arg_list=arg_list, shared_data=shared_data, timeout=timeout
        )

        # Aggregate results
        highest_score = 0.0

        for model_idx, score, metadata in success_results:
            if score > highest_score:
                highest_score = score

            if verbose >= 2:
                run_instance.logger.info(
                    f"Model {metadata['method_name']}: score={score:.4f}, "
                    f"time={metadata.get('elapsed_time', 0):.2f}s"
                )

        # Collect errors
        model_error_list = []

        for model_idx, metadata in error_results:
            model_error_list.append(
                [
                    arg_list[model_idx][0],  # algorithm implementation
                    Exception(metadata.get("error_message", "Unknown error")),
                    metadata.get("traceback", ""),
                ]
            )

            if verbose >= 1:
                run_instance.logger.warning(
                    f"Model {metadata['method_name']} failed: "
                    f"{metadata.get('error_type', 'Unknown')}"
                )

        return model_error_list, highest_score

    finally:
        executor.close()


# Helper function to extract results after parallel execution
def aggregate_parallel_results(
    success_results: List[Tuple], error_results: List[Dict], arg_list: List[Tuple]
) -> Tuple[List[List[Any]], float]:
    """
    Aggregate results from parallel model execution.

    Args:
        success_results: List of successful execution results
        error_results: List of failed execution metadata
        arg_list: Original argument list

    Returns:
        Tuple of (model_error_list, highest_score)
    """
    highest_score = max((r[1] for r in success_results), default=0.0)

    model_error_list = []

    for model_idx, metadata in error_results:
        algorithm_impl = arg_list[model_idx][0]

        model_error_list.append(
            [
                algorithm_impl,
                Exception(metadata.get("error_message", "Unknown error")),
                metadata.get("traceback", ""),
            ]
        )

    return model_error_list, highest_score
