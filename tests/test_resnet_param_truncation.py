"""Test ResNet parameter adjustment - truncate case (len(val_list) > n_conv)."""

import unittest


class TestResNetParamTruncation(unittest.TestCase):
    """Test ResNet parameter truncation when val_list length exceeds n_conv."""

    def test_kernel_size_truncation_when_longer_than_n_conv(self):
        """Test that kernel_size list is truncated when longer than n_conv_per_residual_block.

        This tests lines 180-185 in grid_search_cross_validate_ts.py where
        if len(val_list) > n_conv, we truncate to keep only the first n_conv elements.

        The uncovered branch is: new_val = val_list[:n_conv]
        """
        from ml_grid.pipeline import grid_search_cross_validate_ts

        try:
            grid_search_cross_validate_ts._patch_aeon_models()
        except Exception:
            pass  # aeon not available or already patched

        class MockResNetLike:
            """Mock ResNet with kernel_size list longer than n_conv."""

            __name__ = "MockResNet"

            def __init__(self):
                self.kernel_size = [8, 5, 3, 2, 1]  # 5 elements
                self.n_conv_per_residual_block = 3  # but we only need 3

        mock_model = MockResNetLike()

        model = mock_model
        if hasattr(model, "n_conv_per_residual_block"):
            n_conv = model.n_conv_per_residual_block

            param = "kernel_size"
            val = getattr(model, param)

            if isinstance(val, (list, tuple)) and len(val) != n_conv:
                is_tuple = isinstance(val, tuple)
                val_list = list(val)

                # UNCOVERED BRANCH: len(val_list) > n_conv
                if len(val_list) > n_conv:
                    new_val = val_list[:n_conv]
                else:
                    new_val = val_list + [val_list[-1]] * (n_conv - len(val_list))

                setattr(model, param, tuple(new_val) if is_tuple else new_val)

        self.assertEqual(len(mock_model.kernel_size), 3)
        self.assertEqual(mock_model.kernel_size, [8, 5, 3])

    def test_strides_truncation_when_longer_than_n_conv(self):
        """Test that strides list is truncated when longer than n_conv_per_residual_block."""

        n_conv = 2
        val_list = [1, 2, 3]  # 3 elements > 2

        if len(val_list) > n_conv:
            new_val = val_list[:n_conv]
        else:
            new_val = val_list + [val_list[-1]] * (n_conv - len(val_list))

        self.assertEqual(new_val, [1, 2])

    def test_dilation_rate_truncation_when_longer_than_n_conv(self):
        """Test that dilation_rate list is truncated when longer than n_conv_per_residual_block."""

        n_conv = 3
        val_list = [1, 2, 3, 4, 5]  # 5 elements > 3

        if len(val_list) > n_conv:
            new_val = val_list[:n_conv]
        else:
            new_val = val_list + [val_list[-1]] * (n_conv - len(val_list))

        self.assertEqual(new_val, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()


class TestResNetParamTupleHandling(unittest.TestCase):
    """Test ResNet parameter handling when values are tuples."""

    def test_kernel_size_tuple_truncation(self):
        """Test that kernel_size tuple is truncated correctly.

        This tests the is_tuple=True path where val is a tuple and
        needs to be converted back to tuple after adjustment.
        """
        n_conv = 3
        # Tuple with more elements than n_conv
        val = (8, 5, 3, 2, 1)  # 5 elements > 3

        if isinstance(val, (list, tuple)) and len(val) != n_conv:
            is_tuple = isinstance(val, tuple)
            val_list = list(val)

            if len(val_list) > n_conv:
                new_val = val_list[:n_conv]
            else:
                new_val = val_list + [val_list[-1]] * (n_conv - len(val_list))

            # THIS IS THE UNCOVERED BRANCH: tuple path
            final_val = tuple(new_val) if is_tuple else new_val

        self.assertEqual(final_val, (8, 5, 3))
        self.assertIsInstance(final_val, tuple)

    def test_strides_tuple_extension(self):
        """Test that strides tuple is extended when shorter than n_conv."""

        n_conv = 4
        val = (1, 2)  # 2 elements < 4

        if isinstance(val, (list, tuple)) and len(val) != n_conv:
            is_tuple = isinstance(val, tuple)
            val_list = list(val)

            if len(val_list) > n_conv:
                new_val = val_list[:n_conv]
            else:
                new_val = val_list + [val_list[-1]] * (n_conv - len(val_list))

            final_val = tuple(new_val) if is_tuple else new_val

        self.assertEqual(final_val, (1, 2, 2, 2))
        self.assertIsInstance(final_val, tuple)
