"""Test exception handling for optional imports.

Uses pytest-subprocess to ensure fresh module import.
"""

import subprocess
import sys


def test_torch_import_error_handling():
    """Test that ImportError during torch import is handled gracefully (lines 20-21)."""

    code = """
import sys

# Block torch import after it's been cached elsewhere
if 'torch' in sys.modules:
    del sys.modules['torch']

# Mock the import system to fail on torch
import builtins
orig_import = builtins.__import__

def mock_import(name, *args, **kwargs):
    if name == 'torch':
        raise ImportError(f"No module named '{name}'")
    return orig_import(name, *args, **kwargs)

builtins.__import__ = mock_import

# Import the module - this triggers lines 18-21
from ml_grid.pipeline import grid_search_cross_validate_ts as g

print("SUCCESS: torch import error handled at lines 20-21")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=(
            {**subprocess.os.environ, "COVERAGE_PROCESS_START": ".coveragerc"}
            if subprocess.os.path.exists(".coveragerc")
            else None
        ),
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    assert result.returncode == 0, f"Exit code {result.returncode}"
    assert "SUCCESS" in result.stdout


def test_tf_traceback_error_handling():
    """Test that AttributeError during TF traceback filtering is handled (lines 54-55)."""

    code = """
from unittest.mock import patch

# Mock disable_traceback_filtering to raise AttributeError
with patch('tensorflow.debugging.disable_traceback_filtering',
           side_effect=AttributeError("Mock error")):
    from ml_grid.pipeline import grid_search_cross_validate_ts as g

print("SUCCESS: TF AttributeError handled at lines 54-55")
"""

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    assert result.returncode == 0, f"Exit code {result.returncode}"
    assert "SUCCESS" in result.stdout


if __name__ == "__main__":
    import os

    # Write coverage config
    with open(".coveragerc", "w") as f:
        f.write(
            """[report]
omit = */site-packages/*
"""
        )

    try:
        test_torch_import_error_handling()
        print("\n" + "=" * 60 + "\n")
        test_tf_traceback_error_handling()
        print("\nAll tests passed!")
    finally:
        if os.path.exists(".coveragerc"):
            os.remove(".coveragerc")
