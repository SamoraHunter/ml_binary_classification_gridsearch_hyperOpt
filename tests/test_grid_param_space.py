"""Tests for grid_param_space.py module."""

from ml_grid.util.grid_param_space import Grid


def test_grid_parameter_space_import():
    """Test Grid class can be imported."""
    assert Grid is not None


def test_grid_init_with_default_sample_n():
    """Test Grid initialization with default sample_n value returns instantly."""
    grid = Grid()

    assert hasattr(grid, "grid")
    assert hasattr(grid, "sample_n")


def test_grid_init_with_custom_sample_n():
    """Test Grid initialization with custom sample_n value."""
    grid = Grid(sample_n=10)

    assert grid.sample_n == 10


def test_grid_data_structures_exist():
    """Test that all expected data structures exist in the grid."""
    grid = Grid()

    grid_dict = grid.grid

    assert "resample" in grid_dict
    assert isinstance(grid_dict["resample"], list)

    assert "scale" in grid_dict
    assert isinstance(grid_dict["scale"], list)

    assert "feature_n" in grid_dict
    assert isinstance(grid_dict["feature_n"], list)


def test_grid_resample_options():
    """Test that resample options contain expected values."""
    grid = Grid()

    resample_values = grid.grid["resample"]

    assert "undersample" in resample_values
    assert "oversample" in resample_values
    assert None in resample_values


def test_grid_feature_n_options():
    """Test that feature_n options contain expected values."""
    grid = Grid()

    feature_n_values = grid.grid["feature_n"]

    assert 100 in feature_n_values
    assert 95 in feature_n_values
    assert 50 in feature_n_values


def test_grid_embedding_options():
    """Test embedding-related options exist."""
    grid = Grid()

    assert "use_embedding" in grid.grid
    assert "embedding_method" in grid.grid
    assert "embedding_dim" in grid.grid


def test_settings_list_triggers_c_prod():
    """Test that accessing settings_list property triggers Cartesian product generation."""
    grid = Grid(sample_n=5)

    # Access the lazy property to trigger _c_prod execution
    settings = grid.settings_list

    assert isinstance(settings, list)
    assert len(settings) <= 5
    for setting in settings:
        assert isinstance(setting, dict)


def test_settings_list_iterator_property():
    """Test that settings_list_iterator property works correctly."""
    grid = Grid(sample_n=2)

    iterator = grid.settings_list_iterator

    first_item = next(iterator)
    assert isinstance(first_item, dict)

    second_item = next(iterator)
    assert isinstance(second_item, dict)


def test_sample_n_none_default():
    """Test that sample_n=None defaults to 1000."""
    from ml_grid.util.grid_param_space import Grid

    grid = Grid(sample_n=None)

    assert grid.sample_n == 1000
