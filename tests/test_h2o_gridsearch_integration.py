import pytest
import numpy as np
import h2o
import logging

# Essential ml_grid imports
from ml_grid.pipeline.grid_search_cross_validate import grid_search_crossvalidate
from ml_grid.util.global_params import global_parameters

# Import all H2O model definition classes
from ml_grid.model_classes.h2o_gbm_classifier_class import H2O_GBM_class
from ml_grid.model_classes.h2o_drf_classifier_class import H2O_DRF_class
from ml_grid.model_classes.h2o_gam_classifier_class import H2O_GAM_class
from ml_grid.model_classes.h2o_deeplearning_classifier_class import H2O_DeepLearning_class
from ml_grid.model_classes.h2o_glm_classifier_class import H2O_GLM_class
from ml_grid.model_classes.h2o_naive_bayes_classifier_class import H2O_NaiveBayes_class
from ml_grid.model_classes.h2o_rulefit_classifier_class import H2O_RuleFit_class
from ml_grid.model_classes.h2o_xgboost_classifier_class import H2O_XGBoost_class
from ml_grid.model_classes.h2o_stackedensemble_classifier_class import H2O_StackedEnsemble_class
from ml_grid.model_classes.h2o_classifier_class import H2OAutoMLConfig as H2O_class # AutoML

# A mock class to simulate the main 'pipe' object for integration testing
class MockMlGridObject:
    def __init__(self, X, y, search_strategy='random'):
        self.X_train = X
        self.y_train = y
        self.X_test = X
        self.y_test = y
        self.X_test_orig = X
        self.y_test_orig = y
        self.local_param_dict = {'param_space_size': 'small'}
        self.global_params = global_parameters
        self.base_project_dir = "test_experiments/test_run"
        self.verbose = 0
        self.global_params.cv_folds = 2 # Use 2 folds for faster tests
        self.global_params.verbose = 0
        self.global_params.error_raise = True
        self.global_params.grid_n_jobs = 1 # H2O requires n_jobs=1
        self.global_params.test_mode = True # Skips final CV for speed
        self.global_params.sub_sample_param_space_pct = 1.0

        # Configure search strategy
        if search_strategy == 'random':
            self.global_params.random_grid_search = True
            self.global_params.bayessearch = False
            self.global_params.max_param_space_iter_value = 1
        elif search_strategy == 'grid':
            self.global_params.random_grid_search = False
            self.global_params.bayessearch = False
        elif search_strategy == 'bayes':
            self.global_params.random_grid_search = False
            self.global_params.bayessearch = True
            self.global_params.max_param_space_iter_value = 1

        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.DEBUG)

def _prepare_h2o_param_space(instance, model_class, search_strategy):
    """
    Helper function to prepare and sanitize H2O parameter spaces for testing.
    This centralizes the logic for limiting runtimes and ensuring compatibility.
    """
    param_space = instance.parameter_space

    # Flatten list of dicts into a single dict if necessary
    if isinstance(param_space, list):
        flat_param_space = {}
        for d in param_space:
            flat_param_space.update(d)
        param_space = flat_param_space

    # For grid search, values must be in a list, even if it's a single element.
    # Grid search treats each list element as a separate value to test
    # For random/bayes, we can use small lists for sampling
    
    # 1. For AutoML, force a very short runtime
    if model_class == H2O_class:
        if search_strategy == 'grid':
            param_space['max_runtime_secs'] = [5]
            param_space['max_models'] = [2]
            param_space['sort_metric'] = ["AUC"]
        else:
            param_space['max_runtime_secs'] = [5]
            param_space['max_models'] = [2]
            param_space['sort_metric'] = ["AUC"]

    # 2. For tree-based models, force minimal trees
    if 'ntrees' in param_space:
        if search_strategy == 'grid':
            param_space['ntrees'] = [2]  # Wrap single value in a list
        else:
            param_space['ntrees'] = [2, 3]  # Small list for sampling

    # 3. For Deep Learning, force minimal epochs
    if model_class == H2O_DeepLearning_class and 'epochs' in param_space:
        if search_strategy == 'grid':
            param_space['epochs'] = [1]  # Wrap single value in a list
        else:
            param_space['epochs'] = [1, 2]

    # 4. Add max_runtime_secs to ALL models as safety net
    if 'max_runtime_secs' not in param_space:
        if search_strategy == 'grid':
            param_space['max_runtime_secs'] = [10]  # 10 second timeout
        else:
            param_space['max_runtime_secs'] = [10]

    # 5. For non-Bayesian searches, convert skopt distributions to concrete values
    if search_strategy != 'bayes':
        for key, value in param_space.items():
            if hasattr(value, 'rvs'):  # It's a skopt distribution
                if search_strategy == 'grid':
                    # Single concrete value for grid search, wrapped in a list
                    param_space[key] = [value.rvs(random_state=0)]
                else:  # random search
                    # Convert to small list
                    if hasattr(value, 'categories'):
                        cats = list(value.categories)
                        param_space[key] = cats[:2] if len(cats) > 2 else cats
                    elif hasattr(value, 'low') and isinstance(value.low, int):
                        param_space[key] = [value.low, min(value.low + 1, value.high)]
                    elif hasattr(value, 'low') and isinstance(value.low, float):
                        param_space[key] = [value.low, (value.low + value.high) / 2]
    
    return param_space

# --- PERFORMANCE FIX: Use only fast models ---
# Exclude AutoML and StackedEnsemble as they are too slow/complex for integration tests
H2O_MODELS_TO_TEST = [
    H2O_GLM_class,      # Fast, simple
    H2O_DRF_class,      # Can be fast with limited trees
    H2O_GBM_class,      # Can be fast with limited trees
]

# Optional: Add a separate slow test for comprehensive coverage
H2O_SLOW_MODELS = [
    H2O_DeepLearning_class,
    H2O_GAM_class,
    H2O_NaiveBayes_class,
    H2O_RuleFit_class,
    H2O_XGBoost_class,
]

@pytest.mark.parametrize("search_strategy", ["random", "bayes", "grid"])
@pytest.mark.parametrize("model_class", H2O_MODELS_TO_TEST)
def test_h2o_search_integrations(model_class, search_strategy, synthetic_data, h2o_session_fixture):
    """
    Tests H2O models with all search strategies (Randomized, Bayes, Grid).
    This test is parameterized by both model and search strategy to ensure
    maximum isolation between test runs, which is more stable for H2O.
    """
    X, y = synthetic_data
    
    if model_class == H2O_class:
        instance = model_class(parameter_space_size="small")
    else:
        instance = model_class(X=X, y=y, parameter_space_size="small")

    mock_ml_grid_object = MockMlGridObject(X, y, search_strategy=search_strategy)

    param_space = _prepare_h2o_param_space(
        instance=instance,
        model_class=model_class,
        search_strategy=search_strategy
    )

    # Clean H2O state before each test
    h2o.remove_all()

    result = grid_search_crossvalidate(
        algorithm_implementation=instance.algorithm_implementation,
        parameter_space=param_space,
        method_name=instance.method_name,
        ml_grid_object=mock_ml_grid_object
    )
    
    assert isinstance(result.grid_search_cross_validate_score_result, float)
    
    # Additional cleanup
    h2o.remove_all()


@pytest.mark.slow
@pytest.mark.parametrize("search_strategy", ["random"])  # Only test one strategy for slow models
@pytest.mark.parametrize("model_class", H2O_SLOW_MODELS)
def test_h2o_slow_models(model_class, search_strategy, synthetic_data, h2o_session_fixture):
    """
    Separate test for slower H2O models. Run with: pytest -m slow
    """
    X, y = synthetic_data
    
    instance = model_class(X=X, y=y, parameter_space_size="small")
    mock_ml_grid_object = MockMlGridObject(X, y, search_strategy=search_strategy)

    param_space = _prepare_h2o_param_space(
        instance=instance,
        model_class=model_class,
        search_strategy=search_strategy
    )

    h2o.remove_all()

    result = grid_search_crossvalidate(
        algorithm_implementation=instance.algorithm_implementation,
        parameter_space=param_space,
        method_name=instance.method_name,
        ml_grid_object=mock_ml_grid_object
    )
    
    assert isinstance(result.grid_search_cross_validate_score_result, float)
    h2o.remove_all()