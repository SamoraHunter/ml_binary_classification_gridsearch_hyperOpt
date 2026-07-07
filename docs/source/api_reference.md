# ML Binary Classification Grid Search HyperOpt API Reference

## Module Overview

This module provides a comprehensive machine learning pipeline for binary classification using grid search and hyperparameter optimization techniques. It supports both standard classifiers (scikit-learn, XGBoost, LightGBM, H2O models) and time-series classifiers from the `aeon` library.

The system implements:
- Automated data preprocessing and feature selection
- Cross-validated hyperparameter search with GridSearchCV, RandomizedSearchCV, and Bayesian Optimization
- Model training, evaluation, and result logging
- Comprehensive metrics tracking (AUC, F1, Precision, Recall, Accuracy)
- Time-series specific data processing and model wrapper support

## Module: `ml_grid.util.db_backend`

**Purpose**: Database backend for experiment tracking using SQLite with hierarchical results storage.

### Class: `DatabaseBackend`

Handles database operations including schema initialization and result insertion. Supports three-tier database structure (local run, project folder master, global master).

#### `__init__(self, db_path: str) -> None`

Initializes the database connection and creates tables if they don't exist.

**Parameters**:
- `db_path` (str): Path to the SQLite database file.

#### Methods

##### `insert_result(self, result_data: Dict[str, Any]) -> None`
Inserts a single result record into the database with automatic type conversion for NumPy types and skopt spaces.

**Parameters**:
- `result_data` (Dict[str, Any]): Dictionary containing all result data including metrics and configuration.

##### `_sanitize_value(self, value: Any) -> Any`
Converts numpy types and search space objects to native Python types for SQLite storage.

**Parameters**:
- `value` (Any): Value to sanitize.

**Returns**: Sanitized value suitable for SQLite storage.

---

## Module: `ml_grid.util.validate_parameters`

**Purpose**: Parameter validation helper for hyperparameter search configurations.

### Functions

#### `validate_parameters_helper(algorithm_implementation: Any, parameters: Union[Dict, List[Dict]], ml_grid_object: Optional[Any]) -> Union[Dict, List[Dict]]`

Validates and cleans parameter space for grid/random search.

**Parameters**:
- `algorithm_implementation`: The estimator instance.
- `parameters` (Union[Dict, List[Dict]]): Parameter space definition.
- `ml_grid_object` (Optional[Any]): Pipeline object for context.

**Returns**: Validated parameter space.

#### `_validate_single_param_space(param_dict: Dict, algorithm_name: str) -> Dict`

Validates a single parameter dictionary against scikit-learn API requirements.

**Parameters**:
- `param_dict` (Dict): Single parameter space dictionary.
- `algorithm_name` (str): Algorithm class name for error messages.

**Returns**: Validated parameter dictionary.

---

## Module: `ml_grid.util.debug_print_statements`

**Purpose**: Debug utilities for internal pipeline debugging.

### Class: `debug_print_statements_class`

A utility class for conditional debug printing during pipeline execution.

#### Methods

##### `__init__(self, verbose: int = 0)`
Initializes with specified verbosity level. Lower values produce less output.

##### `print(self, *args, **kwargs)`
Prints arguments if verbose level is enabled.

---

## Module: `ml_grid.util.logger_setup`

**Purpose**: Logger setup utilities for pipeline components.

### Functions

#### `setup_logger(name: str, log_file: Optional[str] = None, to_stdout: bool = True, level: int = logging.INFO) -> logging.Logger`

Configures and returns a logger instance.

**Parameters**:
- `name` (str): Logger name.
- `log_file` (Optional[str]): Path to log file. If None, only stdout.
- `to_stdout` (bool): If True, logs to console.
- `level` (int): Logging level. Defaults to INFO.

**Returns**: Configured logging.Logger instance.

---

## Module: `ml_grid.pipeline.hierarchical_hyperparameter_search`

**Purpose**: Hierarchical hyperparameter search with coarse-to-fine optimization using dynamic space reduction and early stopping.

### Class: `HierarchicalHyperparameterSearch`

Implements multi-stage hyperparameter optimization with progressive refinement. Supports three stages:
1. **Coarse Search**: Broad exploration with minimal evaluations per parameter
2. **Fine Search**: Focused exploitation on promising regions
3. **Refinement**: Detailed optimization of top candidates

#### Methods

##### `__init__(self, initial_param_space: Dict[str, Any], max_total_trials: int = 100, coarse_ratio: float = 0.25, fine_ratio: float = 0.45, refinement_ratio: float = 0.30, logger: logging.Logger = None)`
Initializes hierarchical search configuration.

**Parameters**:
- `initial_param_space` (Dict[str, Any]): Initial parameter space dictionary.
- `max_total_trials` (int): Total number of evaluations planned across all stages.
- `coarse_ratio` (float): Ratio of trials for coarse search stage. Defaults to 0.25.
- `fine_ratio` (float): Ratio of trials for fine search stage. Defaults to 0.45.
- `refinement_ratio` (float): Ratio of trials for refinement stage. Defaults to 0.30.
- `logger` (logging.Logger, optional): Logger instance for progress tracking.

##### `run_hierarchical_search(self, evaluate_fn: callable, max_trials_per_stage: Optional[Dict[str, int]] = None, verbose: bool = True) -> Tuple[SearchResult, Dict[str, List[SearchResult]]]`
Executes hierarchical hyperparameter search across all stages.

**Parameters**:
- `evaluate_fn` (callable): Function to evaluate parameter set. Should return (score, fit_time).
- `max_trials_per_stage` (Optional[Dict[str, int]], optional): Override trial counts per stage.
- `verbose` (bool): Whether to log progress details.

**Returns**:
Tuple of (best_result, all_results_by_stage) where best_result is SearchResult and all_results_by_stage maps stage names to result lists.

---

##### `run_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator`
Executes hierarchical hyperparameter search.

**Parameters**:
- `X_train` (pd.DataFrame): Training features.
- `y_train` (pd.Series): Training labels.

**Returns**: Best estimator from the search process.

---

## Module: `ml_grid.pipeline.test_data_pipeline`

**Purpose**: Test utilities for data pipeline validation.

### Functions

#### `test_data_pipeline(file_name: str, drop_term_list: List[str], experiment_dir: str, base_project_dir: str, local_param_dict: Dict[str, Any]) -> pipe`

Tests the complete data pipeline with sample configuration.

**Parameters**:
- `file_name` (str): Input CSV file path.
- `drop_term_list` (List[str]): Column drop patterns.
- `experiment_dir` (str): Experiment output directory.
- `base_project_dir` (str): Project base directory.
- `local_param_dict` (Dict[str, Any]): Pipeline parameters.

**Returns**: Configured pipe instance.

---

## Module: `ml_grid.results_processing.filters`

**Purpose**: Data filtering utilities for result analysis.

### Functions

#### `filter_outliers(data: pd.DataFrame) -> pd.DataFrame`
Removes outliers from results DataFrame using IQR method.

**Parameters**:
- `data` (pd.DataFrame): Input results data.

**Returns**: Filtered DataFrame without outliers.

#### `filter_low_samples(data: pd.DataFrame, min_samples: int = 100) -> pd.DataFrame`
Filters runs with insufficient sample sizes.

**Parameters**:
- `data` (pd.DataFrame): Input results data.
- `min_samples` (int, optional): Minimum samples threshold. Defaults to 100.

**Returns**: Filtered DataFrame.

---

## Module: `ml_grid.results_processing.plot_*`

**Purpose**: Visualization and plotting functions for results analysis.

### Available Functions

#### `plot_master(experiment_dir) -> pd.DataFrame`
Master plot function that generates comprehensive visualizations for the experiment.

## Module: `ml_grid.pipeline.main`

**Purpose**: Main orchestrator class that executes hyperparameter searches across multiple models.

### Class: `run`

Class that manages the entire hyperparameter search workflow for a list of models.

#### `__init__(self, local_param_dict: Dict[str, Any], **kwargs)`

Initializes the run class with parameter configuration and data pipeline object.

**Parameters**:
- `local_param_dict` (Dict[str, Any]): A dictionary of parameters for the current experimental run, such as `param_space_size`.
- `ml_grid_object` (**kwargs): Optional. The main data pipeline object containing data and model configurations.
- Other **kwargs**: May include `file_name`, `drop_term_list`, `model_class_dict`, `base_project_dir`, `experiment_dir`, and `outcome_var`.

**Key Attributes**:
- `global_params`: Reference to the global parameters singleton instance.
- `ml_grid_object`: The main data pipeline object containing data and model configurations.
- `sub_sample_param_space_pct`: Percentage of parameter space to sample in randomized search.
- `parameter_space_size`: Size of parameter space for base learners (e.g., 'medium', 'xsmall').
- `model_class_list`: List of instantiated model class objects to evaluate.
- `pg_list`: List containing calculated size of parameter grid for each model.
- `mean_parameter_space_val`: Mean size of parameter spaces across all models.
- `sub_sample_parameter_val`: Calculated number of iterations for randomized search.
- `arg_list`: List of argument tuples for grid search function execution.
- `multiprocess`: Flag to enable/disable multiprocessing.
- `model_error_list`: List to store error details during model training.

#### Methods

##### `_prepare_run(self, model_class)`

Prepares arguments for executing grid search on a single model.

**Parameters**:
- `model_class`: A model class reference from the arg_list tuple.

**Returns**:
Tuple containing: (algorithm_implementation, parameter_space, method_name, ml_grid_object, sub_sample_parameter_val, project_score_save_class_instance)

##### `execute_single_model(self, args: Tuple) -> float`

Executes grid search for a single model and returns its score. Designed to be called within a hyperopt objective function.

**Parameters**:
- `args` (Tuple): Argument tuple from arg_list containing model configuration.

**Returns**:
float - The evaluation score of the model.

**Raises**:
- TimeoutError: If model execution exceeds time limit.
- Exception: Other exceptions are caught and logged, returning 0.0 unless error_raise is True.

##### `execute(self) -> Tuple[List[List[Any]], float]`

Executes grid search for each model in the model list. Iterates through configured models and parameter spaces, running cross-validated grid search for each one.

**Returns**:
Tuple containing:
- `model_error_list` (List[List[Any]]): List of model errors with algorithm instance, exception, and traceback.
- `highest_score` (float): Highest score achieved across all successful model runs.

---

## Module: `ml_grid.pipeline.grid_search_cross_validate`

**Purpose**: Core cross-validation implementation for standard tabular models.

### Class: `grid_search_crossvalidate`

Executes cross-validated hyperparameter search with comprehensive error handling and scoring metric collection.

#### `__init__(self, algorithm_implementation: Any, parameter_space: Union[Dict, List[Dict]], method_name: str, ml_grid_object: Any, sub_sample_parameter_val: int = 100, project_score_save_class_instance: Optional[project_score_save_class] = None)`

Initializes and runs cross-validated hyperparameter search.

**Parameters**:
- `algorithm_implementation` (Any): The scikit-learn compatible estimator instance.
- `parameter_space` (Union[Dict, List[Dict]]): Dictionary or list of dictionaries defining the hyperparameter search space.
- `method_name` (str): Name of the algorithm method.
- `ml_grid_object` (Any): Main pipeline object containing all data and parameters for current iteration.
- `sub_sample_parameter_val` (int, optional): Value used to limit number of iterations in randomized search. Defaults to 100.
- `project_score_save_class_instance` (Optional[project_score_save_class], optional): Instance of score saving class. Defaults to None.

**Key Features**:
- Automatic data scaling for SVC models
- Adaptive cross-validation strategy based on dataset characteristics
- H2O model-specific optimizations and error handling
- GPU configuration for TensorFlow-based models
- Timeout enforcement with SIGALRM support

#### Methods

##### `grid_search_cross_validate_score_result` (property)
Returns the final evaluation score from cross-validation.

**Returns**:
float - The best score achieved during hyperparameter search.

## Module: `ml_grid.pipeline.grid_search_cross_validate_ts`

**Purpose**: Optimized cross-validation implementation for time-series models from the aeon library.

### Class: `grid_search_crossvalidate_ts`

A specialized cross-validation wrapper for time-series classifiers from the aeon library, providing numpy array compatibility and model patches for state management issues.

#### `__init__(self, algorithm_implementation: Any, parameter_space: Union[Dict, List[Dict]], method_name: str, ml_grid_object: Any, sub_sample_parameter_val: int = 100, project_score_save_class_instance: Optional[project_score_save_class] = None)`

Initializes time-series cross-validation search.

**Parameters**:
- `algorithm_implementation` (Any): The aeon time-series classifier instance.
- `parameter_space` (Union[Dict, List[Dict]]): Dictionary or list of dictionaries defining the hyperparameter search space.
- `method_name` (str): Name of the algorithm method.
- `ml_grid_object` (Any): Main pipeline object containing all data and parameters for current iteration.
- `sub_sample_parameter_val` (int, optional): Value used to limit number of iterations in randomized search. Defaults to 100.
- `project_score_save_class_instance` (Optional[project_score_save_class], optional): Instance of score saving class. Defaults to None.

**Key Differences from Standard CV**:
- Accepts 3D numpy arrays (N, T, C) format required by aeon deep learning models
- Implements model patches for state management in deep learning models (e.g., _metrics sync)
- Handles padding and shape transformation for deep learning architectures

#### Methods

##### `grid_search_cross_validate_score_result` (property)
Returns the final evaluation score from cross-validation.

**Returns**:
float - The best score achieved during hyperparameter search.

## Module: `ml_grid.pipeline.hyperparameter_search`

**Purpose**: Orchestrates hyperparameter search using various strategies (Grid, Random, Bayesian).

### Class: `HyperparameterSearch`

Manages the selection and execution of hyperparameter search strategies.

#### `__init__(self, algorithm: BaseEstimator, parameter_space: Union[Dict, List[Dict]], method_name: str, global_params: Any = None, sub_sample_pct: int = 100, max_iter: int = 100, ml_grid_object: Any = None, cv: Any = None)`

Initializes the hyperparameter search orchestrator.

**Parameters**:
- `algorithm` (BaseEstimator): The scikit-learn compatible estimator instance.
- `parameter_space` (Union[Dict, List[Dict]]): The hyperparameter search space.
- `method_name` (str): Name of the algorithm.
- `global_params` (Any): Global parameters object containing configuration settings.
- `sub_sample_pct` (int, optional): Percentage of parameter space to sample for randomized search. Defaults to 100.
- `max_iter` (int, optional): Maximum number of iterations for randomized or Bayesian search. Defaults to 100.
- `ml_grid_object` (Any, optional): Main pipeline object containing data and other parameters.
- `cv` (Any, optional): Cross-validation splitting strategy.

#### Methods

##### `run_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator`

Executes the hyperparameter search using GridSearchCV, RandomizedSearchCV, or BayesSearchCV based on configuration.

**Parameters**:
- `X_train` (pd.DataFrame): Training features with reset index.
- `y_train` (pd.Series): Training labels with reset index.

**Returns**:
BaseEstimator - The best estimator found during the search, with `cv_results_` and `best_index_` attributes attached.

---

## Module: `ml_grid.pipeline.data`

**Purpose**: Data processing pipeline with comprehensive preprocessing and feature engineering.

### Class: `pipe`

Represents a single data processing pipeline permutation that reads data, applies cleaning steps, and splits into train/test sets.

#### `__init__(self, file_name: str, drop_term_list: List[str], experiment_dir: str, base_project_dir: str, local_param_dict: Dict[str, Any], param_space_index: int, additional_naming: Optional[str] = None, test_sample_n: int = 0, column_sample_n: int = 0, time_series_mode: bool = False, model_class_dict: Optional[Dict[str, bool]] = None, outcome_var_override: Optional[str] = None, input_df: Optional[pd.DataFrame] = None)`

Initializes the data pipeline with full configuration.

**Parameters**:
- `file_name` (str): Path to the input CSV file.
- `drop_term_list` (List[str]): List of substrings used to identify columns to drop.
- `experiment_dir` (str): Path to parent directory for experimental runs.
- `base_project_dir` (str): Root directory for project where logs and models will be saved.
- `local_param_dict` (Dict[str, Any]): Dictionary of parameters specific to this pipeline run.
- `param_space_index` (int): Index of current parameter space permutation being run.
- `additional_naming` (Optional[str], optional): String to append to log folder names. Defaults to None.
- `test_sample_n` (int, optional): Number of rows to sample from dataset. Defaults to 0.
- `column_sample_n` (int, optional): Number of columns to sample. Defaults to 0.
- `time_series_mode` (bool, optional): Flag to enable time-series specific processing. Defaults to False.
- `model_class_dict` (Optional[Dict[str, bool]], optional): Dictionary specifying which model classes to include. Defaults to None.
- `outcome_var_override` (Optional[str], optional): Specific outcome variable name. Defaults to None.
- `input_df` (Optional[pd.DataFrame], optional): Pre-loaded DataFrame instead of reading from file. Defaults to None.

#### Key Attributes

- `df`: The raw input DataFrame.
- `all_df_columns`: List of all column names from original DataFrame.
- `pertubation_columns`: Columns selected for inclusion based on local_param_dict.
- `drop_list`: Columns identified to be dropped during cleaning.
- `outcome_variable`: Target variable name for the current pipeline.
- `final_column_list`: Final list of feature columns after filtering.
- `X`: Feature matrix after all processing steps.
- `y`: Target variable corresponding to X.
- `X_train`, `X_test`: Training and validation/testing feature sets.
- `y_train`, `y_test`: Training and testing target sets.
- `X_test_orig`, `y_test_orig`: Original, held-out test set for final validation.
- `model_class_list`: List of instantiated model class objects to be evaluated.

#### Methods

##### `_setup_pipeline(self, experiment_dir: str, redirect_stdout: bool)`

Initializes pipeline logger and tracking attributes.

**Parameters**:
- `experiment_dir` (str): Directory where logs will be stored.
- `redirect_stdout` (bool): If True, redirects stdout to log file.

##### `_load_data(self, file_name: str, test_sample_n: int, column_sample_n: int)`

Loads input data from CSV file or sampled subset.

**Parameters**:
- `file_name` (str): Path to input CSV file.
- `test_sample_n` (int): Number of rows to sample.
- `column_sample_n` (int): Number of columns to sample.

##### `_initial_feature_selection(self, local_param_dict, drop_term_list, outcome_var_override)`

Selects features based on configuration parameters.

**Parameters**:
- `local_param_dict`: Dictionary containing feature selection toggles and configuration.
- `drop_term_list`: List of substrings for identifying columns to exclude.
- `outcome_var_override`: Optional specific outcome variable name.

##### `_apply_safety_net(self)`

Ensures at least one feature remains when pruning would leave zero features. Activates fallback mechanism that prioritizes core protected columns, then original perturbed columns, and finally random selection.

**Raises**:
NoFeaturesError: If safety net fails to retain at least one feature.

##### `_create_xy(self)`

Creates feature matrix X and target y with proper index alignment.

##### `_handle_time_series_conversion(self)`

Converts data to 3D format required by aeon time-series classifiers when in time-series mode.

**Raises**:
ImportError/ModuleNotFoundError: If time-series libraries not installed.

##### `_split_data(self)`

Splits processed data into training and testing sets with index alignment checks.

##### `_post_split_cleaning(self)`

Applies cleaning steps post-split to prevent data leakage:
- Column name cleanup
- Constant column removal
- High missing value removal
- Correlated feature removal
- Duplicated column detection

##### `_scale_features(self)`

Applies StandardScaler normalization to features.

##### `_select_features_by_importance(self)`

Selects top features based on importance scores if configured via `feature_n` parameter.

##### `_apply_embeddings(self)`

Applies dimensionality reduction using PCA, SVD, or other embedding methods.

**Parameters**:
Determines method from `local_param_dict.get("embedding_method", "pca")`.

##### `_finalize_pipeline(self)`

Final logging, checks, and model list generation before pipeline completion.

---

## Module: `ml_grid.pipeline.model_class_list`

**Purpose**: Provides the registration system for standard classifiers.

### Functions

#### `get_model_class_list(ml_grid_object: pipe) -> List[Any]`

Generates a list of instantiated model classes based on configuration.

**Parameters**:
- `ml_grid_object` (pipe): Main data pipeline object containing training data and configuration.

**Returns**:
List[Any] - List of instantiated model class objects.

**Features**:
- Automatically disables GPU models when no CUDA is available
- Disables resource-intensive models in CI environments
- Supports automatic activation based on library availability (AutoGluon, TPOT, FLAML, AutoKeras)

#### MODEL_CLASS_MAP

A dictionary mapping configuration names to actual Python class objects for secure instantiation:

**Standard Models**: LogisticRegressionClass, RandomForestClassifierClass, XGBClassifierClass, KNeighborsClassifierClass, SVCClass, MLPClassifierClass, GradientBoostingClassifierClass, CatBoostClassifierClass, GaussianNBClassifierClass, LightGBMClassifierWrapper, AdaBoostClassifierClass

**GPU Models**: KerasClassifierClass, NeuralNetworkClassifier_class

**H2O Models**: H2OAutoMLClass, H2O_GBM_class, H2O_DRF_class, H2O_DeepLearning_class, H2O_GLM_class, H2O_NaiveBayes_class, H2O_RuleFit_class, H2O_XGBoost_class, H2O_StackedEnsemble_class, H2O_GAM_class

---

## Module: `ml_grid.pipeline.model_class_list_ts`

**Purpose**: Provides the registration system for time-series classifiers from aeon library.

### Functions

#### `get_model_class_list_ts(ml_grid_object: pipe) -> List[Any]`

Generates a list of instantiated time-series model classes based on configuration.

**Parameters**:
- `ml_grid_object` (pipe): Main data pipeline object containing training data and global parameters.

**Returns**:
List[Any] - List of instantiated time-series model class objects.

#### TS_MODEL_CLASS_MAP

Dictionary mapping configuration names to aeon classifier classes:

**Time-Series Models**: KNeighborsTimeSeriesClassifier, TimeSeriesForestClassifier, Arsenal, CNNClassifier, InceptionTimeClassifier, HIVECOTEV2, FreshPRINCEClassifier, FCNClassifier, EncoderClassifier, IndividualInceptionClassifier, IndividualTDE, MLPClassifier, MUSE, OrdinalTDE, ResNetClassifier, RocketClassifier, SignatureClassifier, SummaryClassifier, TemporalDictionaryEnsemble, TSFreshClassifier, ElasticEnsemble, Catch22Classifier

---

## Module: `ml_grid.util.global_params`

**Purpose**: Global configuration singleton managing application-wide settings.

### Class: `GlobalParameters`

A singleton class to manage global configuration parameters accessible throughout the application.

#### `__init__(self, debug_level: int = 0, knn_n_jobs: int = -1)`

Initializes the GlobalParameters instance with default values.

**Parameters**:
- `debug_level` (int, optional): Initial debug level. Defaults to 0.
- `knn_n_jobs` (int, optional): Number of jobs for KNN algorithms. Defaults to -1 (use all processors).

#### Key Configuration Attributes

- `verbose` (int): Controls output verbosity during pipeline runs. Higher values produce more detailed logs. Default: 0
- `error_raise` (bool): If True, stops execution on error; if False, continues after logging errors. Default: False
- `random_grid_search` (bool): Use RandomizedSearchCV instead of GridSearchCV. Default: False
- `bayessearch` (bool): Use BayesSearchCV for hyperparameter tuning. Default: True
- `sub_sample_param_space_pct` (float): Percentage of parameter space to sample in randomized search. Default: 0.0005
- `grid_n_jobs` (int): Parallel jobs for hyperparameter search. -1 uses all processors. Default: -1
- `random_state_val` (int): Seed for reproducibility. Default: 1234
- `n_job_model_val` (int): Parallel jobs for model training. Default: -1
- `max_param_space_iter_value` (int): Hard limit on parameter combinations to evaluate. Default: 10
- `store_models` (bool): Whether to save trained models to disk. Default: False
- `metric_list` (Dict[str, Union[str, Callable]]): Dictionary of scoring metrics during CV.
- `model_eval_time_limit` (int): Time limit in seconds for model evaluation. None means no limit.

#### Methods

##### `update_parameters(self, **kwargs)`

Updates global parameters at runtime.

**Parameters**:
**kwargs (Any): Key-value pairs of parameters to update.

**Raises**:
AttributeError: If key is not a valid parameter.

### Singleton Instance

`global_parameters`: The singleton instance accessible throughout the application.

---

## Module: `ml_grid.util.project_score_save`

**Purpose**: Handles experiment logging, result aggregation, and database operations.

### Class: `project_score_save_class`

Manages score logging to CSV files and SQLite databases for experiment tracking.

#### `__init__(self, experiment_dir: str)`

Initializes the score logger and creates log file with headers.

**Parameters**:
- `experiment_dir` (str): Path to the experiment directory where logs will be saved.

**Creates Three Database Backends**:
1. Local Run DB: In timestamped folder (`ml_results.db`)
2. Project Folder Master DB: In experiments base directory (`project_ml_results.db`)
3. Global Master DB: In project root (`global_master_ml_results.db`)

#### Methods

##### `log_to_db(self, result_data: Dict[str, Any])`

Logs results to all three database tiers.

**Parameters**:
- `result_data` (Dict[str, Any]): The full result dictionary including metrics and configuration.

##### `update_score_log(self, ml_grid_object: Any, scores: Dict[str, np.ndarray], best_pred_orig: np.ndarray, current_algorithm: Any, method_name: str, pg: int, start: float, n_iter_v: int, failed: bool, timeout: bool = False)`

Updates the score log with results of a single experiment run.

**Parameters**:
- `ml_grid_object` (Any): Main pipeline object containing all data and parameters.
- `scores` (Dict[str, np.ndarray]): Dictionary of CV scores.
- `best_pred_orig` (np.ndarray): Predictions from best estimator on original test set.
- `current_algorithm` (Any): Best estimator instance from search.
- `method_name` (str): Name of the algorithm method.
- `pg` (int): Size of parameter grid.
- `start` (float): Start time from time.time().
- `n_iter_v` (int): Number of iterations performed.
- `failed` (bool): Flag indicating if run failed.
- `timeout` (bool): Flag indicating if run timed out. Default: False.

---

## Module: `ml_grid.util.bayes_utils`

**Purpose**: Utility functions for Bayesian optimization parameter space calculations.

### Functions

#### `calculate_combinations(parameter_space: Union[Dict[str, Any], List[Dict[str, Any]]], steps: int = 10) -> int`

Approximates the number of parameter combinations for hyperparameter search.

**Parameters**:
- `parameter_space` (Union[Dict[str, Any], List[Dict[str, Any]]]): Single dictionary or list of dictionaries representing parameter space.
- `steps` (int, optional): Granularity for discretizing continuous parameters. Defaults to 10.

**Returns**:
int - Approximate total number of parameter combinations.

#### `is_skopt_space(param_value: Any) -> bool`

Checks if a parameter value is a scikit-optimize space object.

**Parameters**:
- `param_value` (Any): Parameter value to check.

**Returns**:
bool - True if value is Real, Integer, or Categorical from skopt.

---

## Module: `ml_grid.util.param_space`

**Purpose**: Defines standardized parameter space templates for hyperparameter search.

### Class: `ParamSpace`

Provides pre-defined parameter ranges and configurations used across model classes. Supports both standard grid search (with numpy arrays/lists) and Bayesian optimization (with skopt spaces).

#### `__init__(self, size: Optional[str])`

Initializes the ParamSpace with a specified size configuration.

**Parameters**:
- `size` (Optional[str]): The size of parameter space to generate. Valid values are "xsmall", "small", "medium", and "xwide". If None or unrecognized, param_dict will be None.

**Attributes**:
- `param_dict`: Dictionary containing parameter templates categorized by size. Structure depends on bayessearch setting in global_params.

#### Key Parameters (Standard Grid Search Mode)

- `log_small`: Log-spaced values [1e-5, 1e-3, 1e-2] for medium
- `log_med`: Medium logarithmic range for epsilon and learning rates
- `log_large`: Large logarithmic range for C or n_estimators
- `log_large_long`: Extended logarithmic range for aggressive searches
- `log_epoch`: Epoch counts for deep learning models
- `log_zero_one`: Normalized log-spaced values [0.1, 0.5, 1.0] / 10
- `lin_zero_one`: Linearly spaced values [0.0, 0.05, 0.1] / 10

#### Key Parameters (Bayesian Optimization Mode)

Uses skopt.space objects instead of arrays:
- `log_small`: Real(1e-5, 0.1, prior="log-uniform")
- `log_med`, `log_large`: Integer with low/high bounds
- `bool_param`: Categorical([True, False])

#### Methods

##### `get(self, key: str) -> Any`
Retrieves a specific parameter template by key.

**Parameters**:
- `key` (str): The parameter name to retrieve.

**Returns**: Value from param_dict or None if key doesn't exist.
- `bool_param`: List of boolean values [False, True]
- `enum_class_weights`: List of class weight configurations

---

## Module: `ml_grid.results_processing.core`

**Purpose**: Core module for results aggregation and management.

### Class: `ResultsAggregator`

Aggregates ML results from hierarchical folder structures containing experiment runs.

#### `__init__(self, root_folder: str, feature_names_csv: Optional[str] = None)`

Initializes the ResultsAggregator.

**Parameters**:
- `root_folder` (str): Path to master root folder containing experiment run subfolders.
- `feature_names_csv` (Optional[str], optional): Path to CSV file with original feature names for decoding. Defaults to None.

#### Methods

##### `get_available_runs(self) -> List[str]`

Gets list of available run folders by searching for log files recursively.

**Returns**:
List[str] - Sorted list of valid run folder names.

##### `load_single_run(self, timestamp_folder: str) -> pd.DataFrame`

Loads results from a specific timestamped run folder.

**Parameters**:
- `timestamp_folder` (str): Name of the run folder.

**Returns**:
pd.DataFrame - DataFrame containing results for that run.

##### `aggregate_all_runs(self) -> pd.DataFrame`

Aggregates results from all available runs in root folder.

**Returns**:
pd.DataFrame - Single DataFrame with all aggregated results.

##### `load_from_db(self, db_path: str, include_summaries: bool = True, include_models: bool = True) -> pd.DataFrame`

Loads results from SQLite database backend.

**Parameters**:
- `db_path` (str): Path to the SQLite .db file.
- `include_summaries` (bool): Whether to include HYPEROPT_TRIAL_BEST_SCORE records. Default: True.
- `include_models` (bool): Whether to include individual model evaluation records. Default: True.

**Returns**:
pd.DataFrame - Aggregated results from database.

##### `get_summary_stats(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame`

Gets summary statistics for aggregated results.

**Parameters**:
- `data` (Optional[pd.DataFrame], optional): DataFrame to summarize. Uses internal data if None. Defaults to None.

**Returns**:
pd.DataFrame - DataFrame containing descriptive statistics including count of runs, algorithms, and outcomes.

##### `get_outcome_variables(self, data: Optional[pd.DataFrame] = None) -> List[str]`

Gets list of unique outcome variables from data.

**Parameters**: Same as `get_summary_stats`.

**Returns**:
List[str] - Sorted list of unique outcome variable names.

##### `get_data_by_outcome(self, outcome_variable: str, data: Optional[pd.DataFrame] = None) -> pd.DataFrame`

Filters data for a specific outcome variable.

**Parameters**:
- `outcome_variable` (str): Outcome variable to filter by.
- `data` (Optional[pd.DataFrame], optional): DataFrame to filter. Defaults to None.

**Returns**:
pd.DataFrame - DataFrame containing only data for specified outcome.

##### `get_outcome_summary(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame`

Gets summary statistics stratified by outcome variable.

**Parameters**: Same as other methods.

**Returns**:
pd.DataFrame - Multi-index DataFrame with summary statistics for each outcome variable.

### Class: `DataValidator`

A utility class for validating and checking the quality of results data.

#### Methods

##### `validate_data_structure(df: pd.DataFrame) -> Dict[str, Any]`

Validates structure and quality of results DataFrame.

**Parameters**:
- `df` (pd.DataFrame): The DataFrame to validate.

**Returns**:
Dict[str, Any] - Validation report containing:
- total_records
- columns_present
- missing_columns
- data_quality_issues
- outcome_variables
- algorithms
- runs

##### `print_validation_report(validation_report: Dict[str, Any])`

Prints formatted validation report to console.

### Functions

#### `get_clean_data(df: pd.DataFrame, remove_failed: bool = True) -> pd.DataFrame`

Utility function to get clean data for analysis by removing failed runs.

**Parameters**:
- `df` (pd.DataFrame): Input DataFrame.
- `remove_failed` (bool, optional): If True, removes rows where 'failed' column is 1. Defaults to True.

**Returns**:
pd.DataFrame - Cleaned DataFrame.

#### `stratify_by_outcome(df: pd.DataFrame, func: callable, *args: Any, **kwargs: Any) -> Dict[str, Any]`

Applies a function to data stratified by outcome variable.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with 'outcome_variable' column.
- `func` (callable): Function to apply to each outcome's data subset.
- `*args`, `**kwargs`: Arguments to pass to the function.

**Returns**:
Dict[str, Any] - Dictionary with outcome variables as keys and function results as values.

---

## Module: `ml_grid.results_processing.summarize_results`

**Purpose**: Provides methods to create tabular summaries from ML results data.

### Class: `ResultsSummarizer`

Generates concise DataFrame summaries from aggregated results.

#### `__init__(self, data: pd.DataFrame)`

Initializes the summarizer.

**Parameters**:
- `data` (pd.DataFrame): Aggregated results DataFrame.

#### Methods

##### `get_best_model_per_outcome(self, metric: str = "auc") -> pd.DataFrame`

Finds best model for each outcome and expands feature list into boolean columns.

**Parameters**:
- `metric` (str, optional): Performance metric to determine best model. Defaults to 'auc'.

**Returns**:
pd.DataFrame - DataFrame containing best model run for each outcome with additional boolean columns for features.

---

## Module: `ml_grid.results_processing.plot_*`

**Purpose**: Visualization and plotting functions for results analysis.

### Available Functions

#### `plot_master(self, experiment_dir) -> pd.DataFrame`
Master plot function that generates comprehensive visualizations for the experiment.

#### `plot_algorithms(scoring_data)`
Plots algorithm performance comparison charts using box plots or bar charts.

**Parameters**:
- `scoring_data` (pd.DataFrame): Results DataFrame with model scores.

#### `plot_best_model(scoring_data)`
Visualizes best model results per outcome as a ranking chart.

**Parameters**:
- `scoring_data` (pd.DataFrame): Aggregated results data.

#### `plot_distributions(scoring_data)`
Generates distribution plots for scoring metrics using KDE and histograms.

**Parameters**:
- `scoring_data` (pd.DataFrame): Results DataFrame with metric columns.

#### `plot_features(scoring_data, decoded_column_names=None)`
Creates feature importance visualization showing feature selection frequency.

**Parameters**:
- `scoring_data` (pd.DataFrame): Results with decoded features.
- `decoded_column_names` (Optional[List[str]]): Feature name mapping for decoding.

#### `plot_global_importance(global_feature_importance_df)`
Visualizes global feature importance across all runs using bar charts.

**Parameters**:
- `global_feature_importance_df` (pd.DataFrame): DataFrame with feature importance scores.

#### `plot_features_categories(feature_importance_df)`
Plots feature importances categorized by type (bloods, annotations, etc.).

**Parameters**:
- `feature_importance_df` (pd.DataFrame): Feature categorization data.

#### `plot_interactions(scoring_data, decoded_column_names=None)`
Generates interaction plots showing top feature relationships.

**Parameters**:
- `scoring_data` (pd.DataFrame): Results with decoded features.
- `decoded_column_names` (Optional[List[str]]): Feature name mapping.

#### `plot_pipeline_parameters(scoring_data)`
Visualizes pipeline parameter distributions as histograms or bar charts.

**Parameters**:
- `scoring_data` (pd.DataFrame): Results DataFrame with parameter columns.

#### `plot_timeline(experiment_dir)`
Creates timeline visualization of experiment progress and performance over time.

**Parameters**:
- `experiment_dir` (str): Path to experiment directory containing run folders.

---

## Complete Class List

### Standard Classifier Wrappers (in `ml_grid.model_classes/`)

All model wrapper classes follow this interface pattern:
- `algorithm_implementation`: The underlying sklearn estimator or wrapper instance.
- `method_name`: String identifier for the algorithm (e.g., "LogisticRegression").
- `parameter_space`: Dictionary or list of dictionaries defining hyperparameter search space.

#### All Standard Model Classes

**Core Scikit-learn Wrappers**:
- `LogisticRegressionClass`
- `RandomForestClassifierClass`
- `XGBClassifierClass`
- `KNeighborsClassifierClass`
- `SVCClass`
- `MLPClassifierClass`
- `GradientBoostingClassifierClass`
- `AdaBoostClassifierClass`

**Tree-Based Gradient Boosting**:
- `LightGBMClassifierWrapper` - Optimized for large datasets with faster training.
- `CatBoostClassifierClass` - Automatic handling of categorical features.

**Naive Bayes**:
- `GaussianNBClassifierClass`

**Ensemble Methods**:
- `QuadraticDiscriminantAnalysisClass`
- `TabPFNClassifierClass`

**H2O Ensemble Models** (Require H2O library):
- `H2OAutoMLClass` - Automated machine learning with model selection.
- `H2O_GBM_class` - Gradient Boosting Machine for tabular data.
- `H2O_DRF_class` - Distributed Random Forest.
- `H2O_DeepLearning_class` - Neural network classifier.
- `H2O_GLM_class` - Generalized Linear Model.
- `H2O_NaiveBayes_class` - Naive Bayes classifier.
- `H2O_RuleFit_class` - RuleFit ensemble.
- `H2O_XGBoost_class` - H2O's XGBoost implementation.
- `H2O_StackedEnsemble_class` - Stacking ensemble.
- `H2O_GAM_class` - Generalized Additive Model.

**GPU Models**:
- `KerasClassifierClass` - Keras model wrapper with TensorFlow backend.
- `NeuralNetworkClassifier_class` - PyTorch neural network classifier.

**Auto ML Libraries** (require their respective packages):
- `AutoGluonClassifierClass` - AutoML via Amazon AutoGluon.
- `TPOTClassifierClass` - Tree-based pipeline optimization tool.
- `FLAMLClassifierClass` - Fast and lightweight auto ML.
- `AutoKerasClassifierClass` - Keras-based AutoML.

### Time-Series Classifier Wrappers (in `ml_grid.model_classes_time_series/`)

All time-series models wrap the aeon library classifiers with:
- Automatic 3D array conversion (N, T, C) format.
- Model-specific patches for state management issues.
- Padding and shape handling for deep learning architectures.

#### All Time-Series Model Classes

**Classic Time-Series Classifiers**:
- `KNeighborsTimeSeriesClassifier` - k-NN with dynamic time warping distance.
- `TimeSeriesForestClassifier` - Random forest for time series.
- `ElasticEnsemble_class` - Ensemble of elastic distance algorithms.
- `Catch22Classifier_class` - Catch22 feature extraction + classifier.
- `SummaryClassifier_class` - Summary statistics ensemble.
- `OrdinalTDE_class` - Time series dictionary ensemble with ordinal patterns.
- `TemporalDictionaryEnsemble_class` - TDE with misclassification-aware训练.

**Deep Learning Classifiers**:
- `CNNClassifier_class` - Convolutional neural network for time series.
- `InceptionTimeClassifier_class` - Inception-based architecture.
- `IndividualInceptionClassifier_class` - Single inception block.
- `ResNetClassifier_class` - Deep residual network for time series.
- `MLPClassifier_class` - Multilayer perceptron using aeon's pipeline.
- `FCNClassifier_class` - Fully convolutional network.

**Hybrid/Ensemble Methods**:
- `Arsenal_class` - Arsenal ensemble with DTW distance.
- `HIVECOTEV2_class` - Hierarchical Inference from Voting Ensemble of Time Series Classifiers v2.
- `HIVECOTEV1_class` - Original HIVE-COTE implementation.
- `FreshPRINCEClassifier_class` - Feature-based method for time series.

**Transform-based Methods**:
- `MUSE_class` - Multivariate series transformation with sliding windows.
- `RocketClassifier_class` - Random convolutional kernels (RoCKET).
- `SignatureClassifier_class` - Signature-based features from path.

**Specialized**:
- `EncoderClassifier_class` - Encoder using deep learning.
- `TSFreshClassifier_class` - Uses feature extraction (tsfresh) + classifier.

### Utility Classes

#### Model Patching Functions
These functions are used internally to fix state management issues in aeon models:

##### `_patch_aeon_models() -> None`
Applies patches to BaseClassifier and BaseDeepClassifier from aeon:
- Fixes `_metrics` attribute synchronization for Keras-based models.
- Forces ResNet padding='same' to prevent shape mismatches.
- Adjusts kernel_size/strides/dilation_rate when n_conv_per_residual_block is tuned.

##### `_prepare_deep_learning_data(X, min_length=128) -> np.ndarray`
Prepares data for aeon deep learning models:
- Ensures input is numpy array format.
- Converts 2D (N, T) to 3D (N, C=1, T).
- Pads dimensions if below minimum length.
- Transposes to (N, T, C) for Keras channels_last.

---

## Model Classes Reference

### Standard Classifier Wrappers Details

##### LogisticRegressionClass (`logistic_regression_class.py`)
Standard logistic regression with multi-class support. Parameter space includes elasticnet, l1, and l2 penalties with separate configurations.

**Default Parameters**:
- `C`: Log-uniform [1e-5, 1e-2]
- `class_weight`: [None, "balanced"]
- `solver`: ["saga", "newton-cg", "lbfgs"]

##### SVCClass (`svc_class.py`)
Support Vector Classifier with automatic data scaling. Uses StandardScaler if data not pre-scaled.

**Key Features**:
- Data validation and auto-scaling
- Separate parameter spaces for 'ovr' and 'ovo' decision shapes
- Kernel options: rbf, linear, poly, sigmoid

##### XGBClassifierClass (`xgb_classifier_class.py`)
XGBoost wrapper with GPU support configurations.

##### KNeighborsClassifierClass (`knn_classifier_class.py`)
K-Nearest Neighbors classifier with adaptive parameter space adjustments for small datasets.

##### RandomForestClassifierClass
Random Forest wrapper with parallel processing optimization.

##### LightGBMClassifierWrapper
LightGBM wrapper optimized for fast training on large datasets.

##### CatBoostClassifierClass
CatBoost wrapper with automatic handling of categorical features.

##### H2O AutoML & Model Classes (`h5o*.py`)
H2O ensemble models including AutoML, GBM, DRF, DeepLearning, GLM, NaiveBayes, RuleFit, XGBoost, StackedEnsemble, and GAM.

### Time-Series Classifier Wrappers (in `ml_grid.model_classes_time_series/`)

All time-series models wrap the aeon library classifiers and provide:
- Automatic 3D array conversion (N, T, C)
- Model-specific patches for state management issues
- Padding and shape handling for deep learning architectures

#### Key Time-Series Models

##### KNeighborsTimeSeriesClassifier
k-NN classification for time series using dynamic time warping distance.

**Parameters**:
- `n_neighbors` (int): Number of neighbors.
- `distance_params` (dict): Dictionary of distance metric parameters.

##### ResNetClassifier
Deep residual network for time series classification with automatic padding adjustment.

**Key Features**:
- Patching forces padding='same' to prevent shape mismatches
- Adjusts kernel_size, strides, dilation_rate when n_conv_per_residual_block is tuned

##### InceptionTimeClassifier
Inception-based deep learning model for time series classification.

**Parameters**:
- `kernel_size` (tuple): Size of convolutional kernels.
- `n_filters` (int): Number of filters per inception module.
- `depth` (int): Number of inception blocks.

##### MUSE (Multivariate Series)
Multi-variable time series classifier with sliding window transformation and feature aggregation.

**Parameters**:
- `window_sizes` (list): Sizes of sliding windows.
- `feature_processors` (list): Functions to extract features from each window.

##### HIVECOTEV2
Hybrid ensemble of diverse classifiers and transformations. Implements the state-of-the-art template for time series classification.

**Key Components**:
- UCR feature extraction with summary statistics.
- Shapelet transform for pattern discovery.
- Multiple ensemble layers combining different representations.

##### RocketClassifier
Random interval spectral ensemble (ROCKET) for time series classification using convolutional kernels.

**Parameters**:
- `num_kernels` (int): Number of random convolutional kernels.
- `max_kernel_size` (int): Maximum size of convolutional kernel.

##### TDE (Temporal Dictionary Ensemble)
Dictionary-based machine learning for time series classification with word occurrence features.

**Key Features**:
- Sliding window transformation
- Word creation using SAX (Symbolic Aggregate Approximation)
- Feature aggregation via histogram statistics

##### SummaryClassifier
Feature-based ensemble using summary statistics from time series segments.

##### CNNClassifier
Convolutional neural network for time series classification.

##### FCNClassifier
Fully convolutional network for time series.

---

## Additional Model Classes

### Neural Network Classes

#### `NeuralNetworkKerasClassifier` (in `ml_grid.model_classes.NeuralNetworkKerasClassifier.py`)
Wrapper for Keras neural networks with TensorFlow backend.

**Parameters**:
- `input_shape`: Tuple defining input dimensions.
- `n_classes`: Number of output classes.
- `activation`: Activation function for hidden layers.
- `optimizer`: Optimizer instance or string name.

#### `TabPFNClassifierClass`
TabPFN (Transformers for Tabular data with Perturbations) classifier.

**Parameters**:
- `model_dim`: Dimensionality of the transformer model.
- `n_heads`: Number of attention heads.
- `n_layers`: Number of transformer layers.

### AutoML Classes

#### `AutoGluonClassifierClass`
Wrapper for Amazon AutoGluon Tabular.

**Parameters**:
- `time_limit`: Time limit in seconds.
- `num_cpus`: Number of CPUs to use.
- `prefer_longer_training`: Whether to prioritize model quality over training time.

#### `TPOTClassifierClass`
Tree-based pipeline optimization tool with evolutionary algorithms.

**Parameters**:
- `generations`: Number of evolution iterations.
- `population_size`: Number of individuals per generation.
- `max_time_mins`: Maximum evolution time in minutes.

#### `FLAMLClassifierClass`
Fast and Lightweight AutoML using custom search spaces.

**Parameters**:
- `time_budget`: Total time budget in seconds.
- `estimator_list`: List of estimators to consider.

#### `AutoKerasClassifierClass`
AutoML built on Keras with Bayesian optimization.

**Parameters**:
- `max_trials`: Maximum number of different models to try.
- `seed`: Random seed for reproducibility.

---

---

## Utility Functions

### `time_limit(seconds)`

Context manager that enforces time limit on code execution using SIGALRM. Supports nesting by preserving outer timeout values.

**Parameters**:
- seconds (int): Maximum execution time in seconds. If None, 0, negative, or not supported, no timeout is enforced.

**Returns**: Yields a context where code execution must complete within the specified time.

**Raises**:
TimeoutError: If execution exceeds specified time.
Exception: Other exceptions are re-raised after resetting signal handlers.

### `custom_roc_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float`
Custom ROC AUC with single-class handling. Returns np.nan if only one class present or y_pred is None.

**Parameters**:
- `y_true` (np.ndarray): True binary labels.
- `y_pred` (np.ndarray): Target scores or predicted labels.

**Returns**: float - The ROC AUC score or np.nan if undefined.

### `custom_f1_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float`
F1 score wrapper that handles None predictions.

**Parameters**: Same as sklearn.f1_score plus kwargs for precision/recall/beta/hybrid options.
- `y_true`, `y_pred`: Arrays of true and predicted labels.
- `**kwargs`: Additional arguments passed to sklearn's f1_score.

**Returns**: float - The F1 score or raises ValueError if y_pred is None.

### `custom_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float`
Accuracy score wrapper that handles None predictions.

**Parameters**: Same as sklearn.accuracy_score.
- `y_true`, `y_pred`: Arrays of true and predicted labels.
- `**kwargs`: Additional arguments passed to sklearn's accuracy_score.

**Returns**: float - The accuracy score or raises ValueError if y_pred is None.

### `custom_recall_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float`
Recall score wrapper that handles None predictions.

**Parameters**: Same as sklearn.recall_score.
- `y_true`, `y_pred`: Arrays of true and predicted labels.
- `**kwargs`: Additional arguments passed to sklearn's recall_score.

**Returns**: float - The recall score or raises ValueError if y_pred is None.

### `_configure_tensorflow_gpu_env() -> None`
Configures TensorFlow GPU environment by setting XLA_FLAGS for pip-installed CUDA.

This function runs automatically at module import time and finds the CUDA nvcc toolkit location within site-packages to point XLA to the correct libdevice directory.

---

## Error Handling

The pipeline implements comprehensive error handling:

- **TimeoutError**: Catches execution timeouts, continues with 0.0 score unless error_raise=True
- **KeyboardInterrupt**: Treated as timeout in training loops
- **ConvergenceWarning**: Filtered out by default to reduce log noise
- **NoFeaturesError**: Custom exception when all features are pruned
- **IndexError**: Handled during grid search failures

---

## Notes on API Design

1. All public classes have comprehensive docstrings following Google style.
2. Type hints provided for all function parameters and return values.
3. Singleton pattern used for GlobalParameters to ensure consistent configuration.
4. Thread-safe operations with proper parallel job management.
5. Database abstraction allows both file-based CSV logs and SQLite storage.

---

*This API reference was automatically extracted from the ml_binary_classification_gridsearch_hyperOpt source code.*
