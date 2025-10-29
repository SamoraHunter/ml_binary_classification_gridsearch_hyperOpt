import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, ANY, call
import os
import shutil
from sklearn.base import clone
import h2o

# The class to test
from ml_grid.model_classes.H2OBaseClassifier import H2OBaseClassifier, _SHARED_CHECKPOINT_DIR

# A dummy H2O Estimator class for testing
# This class mimics the structure H2OBaseClassifier expects
class MockH2OEstimator:
    # --- FIX: Add export_checkpoints_dir to signature for test_fit_with_checkpointing ---
    def __init__(self, export_checkpoints_dir=None, **kwargs):
        # Store params to verify them later
        self.params = kwargs
        # Mock a model_id, as this is what H2O models have
        self.model_id = f"mock_model_{id(self)}"
        self._model_json = {'output': {'variable_importances': pd.DataFrame()}}

        # If a checkpoint directory is provided, create it to simulate H2O's behavior.
        if export_checkpoints_dir is not None:
            self.params['export_checkpoints_dir'] = export_checkpoints_dir
            os.makedirs(export_checkpoints_dir, exist_ok=True)

    def train(self, x, y, training_frame):
        # Mock the training process. In a real scenario, this would train the model.
        # For our tests, we just need to make sure it's called correctly.
        pass

    def predict(self, test_data):
        # Mock the prediction process
        # Return a mock H2OFrame-like object with a `as_data_frame` method
        mock_pred_frame = MagicMock()
        
        # Create a sample prediction DataFrame
        # The 'predict' column contains the predicted class labels
        # Other columns contain probabilities for each class
        num_rows = test_data.nrows
        predictions = pd.DataFrame({
            'predict': np.random.randint(0, 2, num_rows),
            'p0': np.random.rand(num_rows),
            'p1': 1 - np.random.rand(num_rows),
        })
        mock_pred_frame.as_data_frame.return_value = predictions
        return mock_pred_frame

# Fixtures are reusable components for tests
@pytest.fixture
def sample_data():
    """Provides a sample dataset for training and prediction."""
    X = pd.DataFrame({
        'feature1': np.linspace(0, 100, 20),
        'feature2': np.linspace(100, 0, 20),
        'feature3': [f"cat_{i % 3}" for i in range(20)] # Add a categorical feature
    })
    y = pd.Series([0, 1] * 10, name="outcome")
    return X, y

@pytest.fixture
def classifier_instance():
    """Returns a clean, unfitted instance of H2OBaseClassifier for each test."""
    # We pass the mock estimator class and some dummy hyperparameters
    return H2OBaseClassifier(estimator_class=MockH2OEstimator, seed=42, nfolds=5)

# This is a powerful testing technique where we replace parts of the system
# with mock objects. Here, we mock all interactions with the `h2o` library.
@patch('h2o.H2OFrame')
@patch('h2o.cluster')
@patch('h2o.init')
def test_fit_successful(mock_h2o_init, mock_h2o_cluster, mock_h2o_frame, classifier_instance, sample_data):
    """
    Tests the entire `fit` process to ensure it behaves as expected.
    """
    X, y = sample_data
    
    # --- Setup Mocks ---
    # Mock H2O cluster status to simulate that H2O is running
    mock_h2o_cluster.return_value.is_running.return_value = True
    
    # Mock the H2OFrame constructor to return a mock object with expected properties
    mock_frame_instance = MagicMock()
    mock_frame_instance.types = {'feature1': 'real', 'feature2': 'real', 'feature3': 'enum', 'outcome': 'enum'}
    mock_h2o_frame.return_value = mock_frame_instance

    # --- Action ---
    # Fit the classifier
    classifier_instance.fit(X, y)

    # --- Assertions ---
    # 1. Check that an H2OFrame was created with the correct data
    # We expect one call to H2OFrame with a pandas DataFrame that has X and y concatenated
    pd.testing.assert_frame_equal(mock_h2o_frame.call_args[0][0].drop('outcome', axis=1), X)
    pd.testing.assert_series_equal(mock_h2o_frame.call_args[0][0]['outcome'].reset_index(drop=True), y.astype('category').reset_index(drop=True), check_names=False)

    # 2. Check that the outcome column was converted to a factor (categorical)
    mock_frame_instance.__getitem__.assert_called_with('outcome')
    mock_frame_instance.__getitem__.return_value.asfactor.assert_called_once()

    # 3. Check that the model's `train` method was called
    assert hasattr(classifier_instance, 'model_')
    assert isinstance(classifier_instance.model_, MockH2OEstimator)
    # We can't directly check the call to train because the model object is created inside `fit`,
    # but we can verify the side-effects.

    # 4. Verify that essential attributes were set after fitting
    assert classifier_instance.model_id is not None
    assert hasattr(classifier_instance, 'classes_')
    np.testing.assert_array_equal(classifier_instance.classes_, [0, 1])
    assert hasattr(classifier_instance, 'feature_names_')
    assert classifier_instance.feature_names_ == list(X.columns)
    assert hasattr(classifier_instance, 'feature_types_')
    assert classifier_instance.feature_types_ == {'feature1': 'real', 'feature2': 'real', 'feature3': 'enum'}

@patch('h2o.get_model')
@patch('h2o.H2OFrame')
@patch('h2o.cluster')
def test_predict_successful(mock_h2o_cluster, mock_h2o_frame, mock_h2o_get_model, classifier_instance, sample_data):
    """
    Tests the `predict` method on a pre-fitted classifier.
    """
    X, y = sample_data
    
    # --- Setup: Manually "fit" the classifier by setting the required attributes ---
    classifier_instance.model_id = "fitted_model_123"
    classifier_instance.classes_ = np.unique(y)
    classifier_instance.feature_names_ = list(X.columns)
    classifier_instance.feature_types_ = {'feature1': 'real', 'feature2': 'real', 'feature3': 'enum'}
    
    # --- Setup Mocks ---
    # Mock the H2OFrame that will be created from the input data
    mock_frame_instance = MagicMock()
    mock_frame_instance.nrows = len(X) # This is the crucial fix
    mock_h2o_frame.return_value = mock_frame_instance

    # Mock the model object that `h2o.get_model` will return
    mock_model = MockH2OEstimator()
    mock_h2o_get_model.return_value = mock_model
    
    # Mock H2O cluster status
    mock_h2o_cluster.return_value.is_running.return_value = True

    # --- Action ---
    predictions = classifier_instance.predict(X)

    # --- Assertions ---
    # 1. Check that the model was retrieved from H2O
    mock_h2o_get_model.assert_called_with("fitted_model_123")
    
    # 2. Check that an H2OFrame was created for the prediction data with correct types
    mock_h2o_frame.assert_called_with(X, column_names=list(X.columns), column_types=classifier_instance.feature_types_)

    # 3. Check the output of the prediction
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert predictions.dtype == 'int'

def test_predict_on_unfitted_model_raises_error(classifier_instance, sample_data):
    """
    Ensures that calling `predict` before `fit` raises a RuntimeError.
    """
    X, _ = sample_data
    with pytest.raises(RuntimeError, match="This H2OBaseClassifier instance is not fitted yet"):
        classifier_instance.predict(X)

def test_initialization():
    """
    Tests that the classifier is initialized correctly.
    """
    # Test with a special 'lambda' parameter, which is a Python keyword
    clf = H2OBaseClassifier(estimator_class=MockH2OEstimator, seed=42, lambda_=0.5)
    
    # Check that attributes are set correctly
    assert clf.seed == 42
    assert clf.lambda_ == 0.5
    assert clf.estimator_class == MockH2OEstimator

    # Test that get_params returns the kwargs correctly
    params = clf.get_params()
    assert 'estimator_class' in params
    assert params['seed'] == 42
    assert params['lambda_'] == 0.5

def test_initialization_fails_without_estimator_class():
    """
    Ensures that the classifier cannot be initialized without a valid estimator class.
    """
    with pytest.raises(ValueError, match="estimator_class is a required parameter"):
        H2OBaseClassifier(estimator_class=None)
    with pytest.raises(ValueError, match="estimator_class is a required parameter"):
        H2OBaseClassifier(estimator_class="not_a_class")

def test_cloning_preserves_params_but_not_fitted_state(classifier_instance, sample_data):
    """
    Tests scikit-learn compatibility by cloning the estimator.
    A cloned estimator should have the same parameters but should not be fitted.
    """
    X, y = sample_data
    
    # --- Setup: Fit the original classifier ---
    # We need to mock the h2o environment for fit to work
    with patch('h2o.H2OFrame'), patch('h2o.cluster'), patch('h2o.init'):
        classifier_instance.fit(X, y)
    
    # Ensure it's fitted
    assert hasattr(classifier_instance, 'model_id')
    assert classifier_instance.model_id is not None

    # --- Action: Clone the fitted classifier ---
    cloned_clf = clone(classifier_instance)

    # --- Assertions ---
    # 1. The clone should have the same parameters from get_params()
    original_params = classifier_instance.get_params()
    cloned_params = cloned_clf.get_params()
    assert cloned_params['estimator_class'] == original_params['estimator_class']
    assert cloned_params['seed'] == original_params['seed']
    assert cloned_params['nfolds'] == original_params['nfolds']

    # 2. The clone should NOT be fitted (no fitted attributes)
    assert not hasattr(cloned_clf, 'model_id')
    assert cloned_clf.model_ is None
    assert cloned_clf.classes_ is None
    assert cloned_clf.feature_names_ is None

    # 3. The original should still be fitted
    assert hasattr(classifier_instance, 'model_id')

def test_input_validation_raises_errors(classifier_instance, sample_data):
    """
    Tests the internal `_validate_input_data` method for various failure cases.
    """
    X, y = sample_data
    
    # Case 1: y contains NaNs
    y_with_nan = y.copy().astype(float)
    y_with_nan.iloc[5] = np.nan
    with pytest.raises(ValueError, match="Target variable y contains NaN values"):
        classifier_instance._validate_input_data(X, y_with_nan)

    # Case 2: X and y have different lengths
    with pytest.raises(ValueError, match="X and y must have same length"):
        classifier_instance._validate_input_data(X.head(5), y)
        
    # Case 3: y has only one class
    y_one_class = pd.Series([0] * len(y), name="outcome")
    with pytest.raises(ValueError, match="y must have at least 2 classes"):
        classifier_instance._validate_input_data(X, y_one_class)

    # Case 4: X contains NaNs
    X_with_nan = X.copy()
    X_with_nan.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="Input data contains NaN values"):
        classifier_instance._validate_input_data(X_with_nan, y)

@patch('h2o.H2OFrame')
@patch('h2o.cluster')
@patch('h2o.init')
def test_fit_with_checkpointing(mock_h2o_init, mock_h2o_cluster, mock_h2o_frame, sample_data):
    """
    Tests that the underlying estimator is created with the `export_checkpoints_dir`
    parameter pointing to the shared directory.
    """
    X, y = sample_data

    # Clean up any previous test runs
    if os.path.exists(_SHARED_CHECKPOINT_DIR):
        shutil.rmtree(_SHARED_CHECKPOINT_DIR)
    
    # The H2OBaseClassifier now unconditionally adds the checkpoint directory
    classifier = H2OBaseClassifier(estimator_class=MockH2OEstimator)
    
    # --- Mocks ---
    mock_h2o_cluster.return_value.is_running.return_value = True
    mock_frame_instance = MagicMock()
    mock_frame_instance.types = {'feature1': 'real', 'feature2': 'real', 'feature3': 'enum', 'outcome': 'enum'}
    mock_h2o_frame.return_value = mock_frame_instance
    
    # --- Action ---
    classifier.fit(X, y)
    
    # --- Assertions ---
    # 1. Check that the estimator was passed the checkpoint parameter
    # (Our MockH2OEstimator stores its init kwargs in `self.params`)
    assert 'export_checkpoints_dir' in classifier.model_.params
    assert classifier.model_.params['export_checkpoints_dir'] == _SHARED_CHECKPOINT_DIR
    
    # 3. Check that the shared directory was created
    assert os.path.exists(_SHARED_CHECKPOINT_DIR)

    # 4. Clean up the created directory
    shutil.rmtree(_SHARED_CHECKPOINT_DIR)