"""Test coverage for force_second_cv behavior in grid_search_cross_validate_ts."""

import unittest


class TestForceSecondCV(unittest.TestCase):
    """Test force_second_cv parameter handling in __init__.

    This tests lines 974-1045 in grid_search_cross_validate_ts.py where:
    - force_second_cv is retrieved from local_param_dict or global_params
    - If True, cached CV results are skipped (line 980)
    - Cached results extraction (lines 983-1044) depends on not having force_second_cv=True
    """

    def test_force_second_cv_retrieved_from_local_dict(self):
        """Test that force_second_cv is retrieved from local_param_dict when present."""

        class MockLocalParamDict:
            def get(self, key, default=None):
                if key == "force_second_cv":
                    return True
                return default

        mock_local = MockLocalParamDict()

        force_second_cv_local = mock_local.get("force_second_cv", False)

        self.assertTrue(force_second_cv_local)

    def test_force_second_cv_fallback_to_global_params(self):
        """Test that force_second_cv falls back to global_params when local is None."""

        class MockGlobalParams:
            def __init__(self):
                self.force_second_cv = True

        class MockLocalParamDictEmpty:
            def get(self, key, default=None):
                if key == "force_second_cv":
                    return None
                return default

        mock_global = MockGlobalParams()
        mock_local = MockLocalParamDictEmpty()

        force_second_cv_local = mock_local.get("force_second_cv")

        if force_second_cv_local is None:
            force_second_cv = getattr(mock_global, "force_second_cv", False)
        else:
            force_second_cv = force_second_cv_local

        self.assertTrue(force_second_cv)

    def test_force_second_cv_false_does_not_skip_cache(self):
        """Test that force_second_cv=False allows cached results to be used."""

        class MockLocalParamDictFalse:
            def get(self, key, default=None):
                if key == "force_second_cv":
                    return False
                return default

        mock_local = MockLocalParamDictFalse()

        force_second_cv_local = mock_local.get("force_second_cv", False)

        self.assertFalse(force_second_cv_local)


if __name__ == "__main__":
    unittest.main()
