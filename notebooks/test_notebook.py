import os
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_notebook():
    # Get the directory where this test file is located
    test_dir = Path(__file__).parent

    # Change to project root (test_dir.parent) before executing notebook
    # This ensures all relative paths in the notebook resolve correctly
    original_cwd = os.getcwd()
    os.chdir(test_dir.parent)

    try:
        with open("notebooks/unit_test_synthetic.ipynb") as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="ml_grid_env")
        ep.preprocess(nb)  # Will raise an error if the notebook fails
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
