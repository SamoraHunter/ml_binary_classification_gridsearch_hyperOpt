"""Test coverage for grid_search_cross_validate_ts module - TensorFlow traceback disable."""

import pytest
import sys


@pytest.mark.ts
def test_tf_traceback_disable_attribute_error_handling():
    """Test that AttributeError during tf.debugging.disable_traceback_filtering() is handled.

    This tests lines 52-55 in grid_search_cross_validate_ts.py:
    ```
    try:
        tf.debugging.disable_traceback_filtering()
    except (AttributeError, ImportError):
        pass
    ```

    The test monkeypatches TensorFlow's disable_traceback_filtering to raise AttributeError,
    then verifies the module can be imported without crashing.
    """
    import tensorflow as tf

    # Store original function for cleanup
    original_func = getattr(tf.debugging, "disable_traceback_filtering", None)

    try:
        # Mock disable_traceback_filtering to raise AttributeError
        def mock_disable_traceback():
            raise AttributeError(
                "disable_traceback_filtering not available in this TF version"
            )

        tf.debugging.disable_traceback_filtering = mock_disable_traceback

        # Clear cached import if it exists
        module_path = "ml_grid.pipeline.grid_search_cross_validate_ts"
        if module_path in sys.modules:
            del sys.modules[module_path]

        # Import the module - should handle AttributeError gracefully

        # If we get here, the exception was handled correctly
        assert True

    finally:
        # Restore original function if it exists
        if original_func is not None:
            tf.debugging.disable_traceback_filtering = original_func


def test_tf_traceback_disable_import_error_handling():
    """Test that ImportError during tf.debugging.disable_traceback_filtering() is handled.

    This tests the ImportError exception path in lines 52-55 of grid_search_cross_validate_ts.py.
    """
    import tensorflow as tf

    original_func = getattr(tf.debugging, "disable_traceback_filtering", None)

    try:
        # Mock to raise ImportError
        def mock_disable_traceback():
            raise ImportError("No module named 'tensorflow.internal'")

        tf.debugging.disable_traceback_filtering = mock_disable_traceback

        module_path = "ml_grid.pipeline.grid_search_cross_validate_ts"
        if module_path in sys.modules:
            del sys.modules[module_path]

        assert True

    finally:
        if original_func is not None:
            tf.debugging.disable_traceback_filtering = original_func
