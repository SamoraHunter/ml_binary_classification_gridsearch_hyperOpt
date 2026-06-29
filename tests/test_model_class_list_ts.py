"""Test suite for the ml_grid.pipeline.model_class_list_ts module."""

import sys
from unittest.mock import MagicMock


def test_get_model_class_list_ts_none_dict():
    """Tests that when model_class_dict is None, an empty list is returned."""
    mock_modules = _create_mock_modules()
    sys.modules.update(mock_modules)

    from ml_grid.pipeline.model_class_list_ts import get_model_class_list_ts

    class MockPipe:
        def __init__(self):
            self.model_class_dict = None

    mock_pipe = MockPipe()
    result = get_model_class_list_ts(mock_pipe)

    assert isinstance(result, list)
    assert len(result) == 0


def test_get_model_class_list_ts_empty_dict():
    """Tests that when model_class_dict is an empty dict, an empty list is returned."""
    mock_modules = _create_mock_modules()
    sys.modules.update(mock_modules)

    from ml_grid.pipeline.model_class_list_ts import get_model_class_list_ts

    class MockPipe:
        def __init__(self):
            self.model_class_dict = {}

    mock_pipe = MockPipe()
    result = get_model_class_list_ts(mock_pipe)

    assert isinstance(result, list)
    assert len(result) == 0


def test_get_model_class_list_ts_all_false():
    """Tests that when all model includes are False, an empty list is returned."""
    mock_modules = _create_mock_modules()
    sys.modules.update(mock_modules)

    from ml_grid.pipeline.model_class_list_ts import get_model_class_list_ts

    class MockPipe:
        def __init__(self):
            self.model_class_dict = {
                "KNeighborsTimeSeriesClassifier": False,
                "TimeSeriesForestClassifier": False,
            }

    mock_pipe = MockPipe()
    result = get_model_class_list_ts(mock_pipe)

    assert isinstance(result, list)
    assert len(result) == 0


def test_get_model_class_list_ts_single_include():
    """Tests that when a single model is set to True, it is instantiated."""
    mock_modules = _create_mock_modules()
    sys.modules.update(mock_modules)

    from ml_grid.pipeline.model_class_list_ts import get_model_class_list_ts

    class MockPipe:
        def __init__(self):
            self.model_class_dict = {
                "KNeighborsTimeSeriesClassifier": True,
            }

    mock_pipe = MockPipe()
    result = get_model_class_list_ts(mock_pipe)

    assert isinstance(result, list)
    assert len(result) > 0


def test_get_model_class_list_ts_unknown_model():
    """Tests that when a model class is not found in TS_MODEL_CLASS_MAP, it logs a warning."""
    # Remove the module from sys.modules if already imported to ensure fresh import
    modules_to_remove = [
        "ml_grid.model_classes_time_series.KNeighborsTimeSeriesClassifier_module",
        "ml_grid.pipeline.model_class_list_ts",
    ]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)

    mock_modules = _create_mock_modules()

    # Create a mock that has the KNN class
    failing_module = MagicMock()

    class FailingKNNClass:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Failed to instantiate")

    failing_module.KNeighborsTimeSeriesClassifier_class = FailingKNNClass
    mock_modules[
        "ml_grid.model_classes_time_series.KNeighborsTimeSeriesClassifier_module"
    ] = failing_module

    sys.modules.update(mock_modules)

    from ml_grid.pipeline.model_class_list_ts import get_model_class_list_ts

    class MockPipe:
        def __init__(self):
            self.model_class_dict = {
                "NonExistentModel": True,  # Model not in TS_MODEL_CLASS_MAP
            }

    mock_pipe = MockPipe()
    result = get_model_class_list_ts(mock_pipe)

    assert isinstance(result, list)
    assert len(result) == 0


def test_get_model_class_list_ts_instantiation_failure():
    """Tests that when model instantiation fails, it logs an error and continues."""
    # Remove the module from sys.modules if already imported to ensure fresh import
    modules_to_remove = [
        "ml_grid.model_classes_time_series.KNeighborsTimeSeriesClassifier_module",
        "ml_grid.pipeline.model_class_list_ts",
    ]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)

    mock_modules = _create_mock_modules()

    # Create a mock KNN class that raises an exception on instantiation
    class FailingKNNClass:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Failed to instantiate")

    # Replace the module with our modified version that has the failing class
    failing_module = MagicMock()
    failing_module.KNeighborsTimeSeriesClassifier_class = FailingKNNClass
    mock_modules[
        "ml_grid.model_classes_time_series.KNeighborsTimeSeriesClassifier_module"
    ] = failing_module

    sys.modules.update(mock_modules)

    from ml_grid.pipeline.model_class_list_ts import get_model_class_list_ts

    class MockPipe:
        def __init__(self):
            self.model_class_dict = {
                "KNeighborsTimeSeriesClassifier": True,
            }

    mock_pipe = MockPipe()
    result = get_model_class_list_ts(mock_pipe)

    assert isinstance(result, list)
    assert len(result) == 0


def _create_mock_modules():
    """Create mock modules for all aeon time-series classifiers."""
    mock_modules = {}
    module_paths = [
        "ml_grid.model_classes_time_series.ArsenalClassifier_module",
        "ml_grid.model_classes_time_series.Catch22Classifer_module",
        "ml_grid.model_classes_time_series.CNNClassifier_module",
        "ml_grid.model_classes_time_series.ContractableBOSSClassifier_module",
        "ml_grid.model_classes_time_series.elasticEnsembleClassifier_module",
        "ml_grid.model_classes_time_series.EncoderClassifier_module",
        "ml_grid.model_classes_time_series.FCNClassifier_module",
        "ml_grid.model_classes_time_series.FreshPRINCEClassifier_module",
        "ml_grid.model_classes_time_series.HIVECOTEV1Classifier_module",
        "ml_grid.model_classes_time_series.HIVECOTEV2Classifier_module",
        "ml_grid.model_classes_time_series.InceptionTimeClassifer_module",
        "ml_grid.model_classes_time_series.IndividualInceptionClassifier_module",
        "ml_grid.model_classes_time_series.InidividualTDEClassifier_module",
        "ml_grid.model_classes_time_series.KNeighborsTimeSeriesClassifier_module",
        "ml_grid.model_classes_time_series.MLPClassifier_module",
        "ml_grid.model_classes_time_series.MUSEClassifier_module",
        "ml_grid.model_classes_time_series.OrdinalTDEClassifier_module",
        "ml_grid.model_classes_time_series.ResNetClassifier_module",
        "ml_grid.model_classes_time_series.rocketClassifier_module",
        "ml_grid.model_classes_time_series.SignatureClassifier_module",
        "ml_grid.model_classes_time_series.SummaryClassifier_module",
        "ml_grid.model_classes_time_series.TemporalDictionaryEnsembleClassifier_module",
        "ml_grid.model_classes_time_series.TimeSeriesForestClassifier_module",
        "ml_grid.model_classes_time_series.TSFreshClassifier_module",
    ]

    for mod_name in module_paths:
        mock_mod = MagicMock()
        parts = mod_name.split("_module")[0].split(".")[-1]

        if parts.endswith("Classifier"):
            class_name = parts.replace("Module", "") + "_class"
        elif parts == "Arsenal":
            class_name = "Arsenal_class"
        else:
            class_name = parts + "_class"

        setattr(mock_mod, class_name[:-6], MagicMock())
        mock_modules[mod_name] = mock_mod

    return mock_modules
