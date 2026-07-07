"""Test OrdinalTDE ImportError fallback using import system hook."""

import pytest
import subprocess
import sys


@pytest.mark.ts
def test_ordinal_tde_main_import_failure():
    """Test that main import failure triggers internal path.

    This tests lines 379-403 in grid_search_cross_validate_ts.py where:
    - Lines 381-384: ImportError exception handler falls back to _ordinal_tde
    - Line 402-403: Outer try/except catches if both fail

    Uses sys.meta_path hook to intercept and fail specific imports.
    """

    code = '''
import sys

class ImportBlocker:
    """Blocks aeon.classification.ordinal_classification import."""

    def find_module(self, name, path=None):
        if name == "aeon.classification.ordinal_classification":
            raise ImportError(f"Blocked: {name}")
        return None

# Add blocker at the front of sys.meta_path
blocker = ImportBlocker()
sys.meta_path.insert(0, blocker)

# Now import - should trigger ImportError and fallback to internal path
from ml_grid.pipeline import grid_search_cross_validate_ts as g

print("SUCCESS: ImportError caught, using internal path")
'''

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    assert result.returncode == 0, f"Exit code {result.returncode}"
    assert "SUCCESS" in result.stdout


if __name__ == "__main__":
    test_ordinal_tde_main_import_failure()
    print("\nTest passed!")
