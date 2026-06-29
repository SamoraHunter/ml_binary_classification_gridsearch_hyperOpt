"""Tests for logger_setup_notebook module."""

import sys
from unittest.mock import patch


def test_setup_logger_basic_functionality():
    """Test that setup_logger returns a valid logger object."""
    from ml_grid.util import logger_setup_notebook

    with patch.object(sys, "settrace"):
        mock_logger = logger_setup_notebook.setup_logger(notebook_verbose=False)

        assert mock_logger is not None
        assert hasattr(mock_logger, "info")
        assert hasattr(mock_logger, "debug")


def test_setup_logger_with_verbose():
    """Test that setup_logger handles notebook_verbose parameter."""
    from ml_grid.util import logger_setup_notebook

    with patch.object(sys, "settrace"):
        mock_logger = logger_setup_notebook.setup_logger(notebook_verbose=True)

        assert mock_logger is not None
        assert hasattr(mock_logger, "info")
