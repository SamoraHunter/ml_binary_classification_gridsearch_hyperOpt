"""
Pytest configuration file for shared fixtures.

This file makes fixtures available to all test files in this directory
and its subdirectories without needing to import them.
"""

import pytest
import pandas as pd
import numpy as np
import h2o

# Add the project root directory to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def h2o_session_fixture():
    """Initializes H2O once per test session for stability and speed."""
    h2o.init(nthreads=1, log_level="FATA")
    yield
    h2o.shutdown(prompt=False)

@pytest.fixture(scope="session")
def synthetic_data():
    """Provides a simple, reusable dataset for testing classifiers."""
    X = pd.DataFrame(np.random.rand(50, 3), columns=['f1', 'f2', 'f3'])
    y = pd.Series(np.random.randint(0, 2, 50), name="outcome")
    return X, y