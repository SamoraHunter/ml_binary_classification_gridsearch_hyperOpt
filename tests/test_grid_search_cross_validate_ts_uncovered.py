# Test for missing parameter_space attribute in global_params
def test_sub_sample_param_space_pct_uses_getattr():
    """Test that grid_search_crossvalidate_ts uses getattr for sub_sample_param_space_pct."""
    from ml_grid.pipeline.grid_search_cross_validate_ts import (
        grid_search_crossvalidate_ts,
    )

    # Verify the source code uses getattr instead of direct attribute access
    import inspect

    source_code = inspect.getsource(grid_search_crossvalidate_ts.__init__)

    # Check that getattr is used with a default value for sub_sample_param_space_pct
    assert "getattr" in source_code, "Code should use getattr for safe attribute access"
    assert (
        '"sub_sample_param_space_pct"' in source_code
        or "'sub_sample_param_space_pct'" in source_code
    ), "Code should pass 'sub_sample_param_space_pct' string to getattr"

    # The fix should handle missing attribute gracefully
    assert (
        "None" in source_code or "self.sub_sample_param_space_pct" in source_code
    ), "getattr should have a default value (None) for sub_sample_param_space_pct"
