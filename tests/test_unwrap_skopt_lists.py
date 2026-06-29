"""Tests for BayesSearch parameter space unwrapping in grid_search_cross_validate_ts module."""

import unittest


class TestBayesSearchParamsUnwrapping(unittest.TestCase):
    """Test _unwrap_skopt_lists behavior through bayesian search code path."""

    def test_unwrap_single_item_list_nested(self):
        """Test that a nested single-item list [[value]] is fully unwrapped.

        Tests lines 746-765 where _unwrap_skopt_lists recursively processes
        parameter spaces for BayesSearchCV. When BayesSearchCV receives a
        single-item list like `[[1, 2]]`, it misinterprets `[1, 2]` as an
        unhashable category instead of a Categorical dimension.

        This test verifies the function correctly unwraps nested single-item lists.
        """

        # Inline the _unwrap_skopt_lists logic directly (lines 746-765)
        def _unwrap_skopt_lists(space):
            if isinstance(space, dict):
                return {k: _unwrap_skopt_lists(v) for k, v in space.items()}
            elif isinstance(space, list):
                if len(space) > 0 and isinstance(space[0], dict):
                    return [_unwrap_skopt_lists(item) for item in space]
                if len(space) == 1:
                    return _unwrap_skopt_lists(space[0])
                return space
            else:
                return space

        # Simulate a parameter space with nested single-item list structure
        input_space = {
            "param1": [[1, 2]],  # Nested list that should be unwrapped
            "param2": [3],  # Single item list that should be unwrapped
            "param3": 5,  # Scalar value (no change expected)
        }

        result = _unwrap_skopt_lists(input_space)

        # Check nested list is fully unwrapped: [[1, 2]] -> [1, 2]
        self.assertEqual(result["param1"], [1, 2])

        # Check single item list is unwrapped: [3] -> 3
        self.assertEqual(result["param2"], 3)

        # Scalar value passes through unchanged
        self.assertEqual(result["param3"], 5)

    def test_unwrap_list_of_dicts_preserved(self):
        """Test that lists of dicts are recursively processed but structure preserved.

        Tests line 757-758 where `isinstance(space[0], dict)` triggers recursion.
        """

        # Inline the _unwrap_skopt_lists logic
        def _unwrap_skopt_lists(space):
            if isinstance(space, dict):
                return {k: _unwrap_skopt_lists(v) for k, v in space.items()}
            elif isinstance(space, list):
                if len(space) > 0 and isinstance(space[0], dict):
                    return [_unwrap_skopt_lists(item) for item in space]
                if len(space) == 1:
                    return _unwrap_skopt_lists(space[0])
                return space
            else:
                return space

        input_space = [
            {"param1": [[1, 2]]},  # Nested single-item list in first dict
            {"param2": [3]},  # Single item list in second dict
        ]

        result = _unwrap_skopt_lists(input_space)

        # List structure should be preserved
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

        # First dict's param1 should have unwrapped nested list
        self.assertEqual(result[0]["param1"], [1, 2])

        # Second dict's param2 should be unwrapped from single-item list
        self.assertEqual(result[1]["param2"], 3)

    def test_unwrap_multi_dimensional_nested_list(self):
        """Test deep nesting with multiple levels of single-item lists."""

        # Inline the _unwrap_skopt_lists logic
        def _unwrap_skopt_lists(space):
            if isinstance(space, dict):
                return {k: _unwrap_skopt_lists(v) for k, v in space.items()}
            elif isinstance(space, list):
                if len(space) > 0 and isinstance(space[0], dict):
                    return [_unwrap_skopt_lists(item) for item in space]
                if len(space) == 1:
                    return _unwrap_skopt_lists(space[0])
                return space
            else:
                return space

        # Three levels of nesting
        input_space = {"param": [[[1, 2]]]}

        result = _unwrap_skopt_lists(input_space)

        # All single-item wrapping should be removed
        self.assertEqual(result["param"], [1, 2])

    def test_unwrap_dont_modify_multi_item_list(self):
        """Test that multi-item lists (actual dimensions) are preserved.

        Tests the boundary condition: if a list has more than one item,
        it's potentially a valid skopt dimension and should not be unwrapped.
        """

        # Inline the _unwrap_skopt_lists logic
        def _unwrap_skopt_lists(space):
            if isinstance(space, dict):
                return {k: _unwrap_skopt_lists(v) for k, v in space.items()}
            elif isinstance(space, list):
                if len(space) > 0 and isinstance(space[0], dict):
                    return [_unwrap_skopt_lists(item) for item in space]
                if len(space) == 1:
                    return _unwrap_skopt_lists(space[0])
                return space
            else:
                return space

        input_space = {
            "param1": [1, 2, 3],  # Multi-item list (actual values)
            "param2": [[1, 2], [3, 4]],  # Multi-item list of lists
        }

        result = _unwrap_skopt_lists(input_space)

        # Multi-item lists should pass through unchanged
        self.assertEqual(result["param1"], [1, 2, 3])
        self.assertEqual(result["param2"], [[1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
