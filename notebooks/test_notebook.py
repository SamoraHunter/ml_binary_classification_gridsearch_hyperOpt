import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_notebook():
    with open("notebooks/unit_test_synthetic.ipynb") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="ml_grid_env")
    ep.preprocess(nb)  # Will raise an error if the notebook fails
