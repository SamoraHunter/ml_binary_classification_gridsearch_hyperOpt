"""
Unit tests for loading and validating the structure of the config.yml file.

These tests ensure that the configuration file exists, is well-formed,
and contains all the necessary sections and keys for the project to run.
"""

import pytest
import yaml
from pathlib import Path

# Define paths to the configuration files
CONFIG_SINGLE_RUN_PATH = Path("notebooks/config_single_run.yml")
CONFIG_HYPEROPT_PATH = Path("notebooks/config_hyperopt.yml")


@pytest.fixture(scope="module", params=[CONFIG_SINGLE_RUN_PATH, CONFIG_HYPEROPT_PATH], ids=["single_run", "hyperopt"])
def config_file(request):
    """
    A pytest fixture that parametrizes tests to run against both config files.
    """
    config_path = request.param
    if not config_path.exists():
        pytest.fail(f"Configuration file not found at {config_path.resolve()}")
    with open(config_path, 'r') as f:
        try:
            config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Failed to parse YAML file {config_path.resolve()}: {e}")
    return config_path.name, config_data


def test_config_is_valid_yaml(config_file):
    """Tests if the loaded config is a dictionary, indicating valid YAML."""
    config_name, config_data = config_file
    assert isinstance(config_data, dict), f"{config_name} is not a valid YAML dictionary."


def test_top_level_sections_exist(config_file):
    """Tests for the presence of all expected top-level sections."""
    config_name, config_data = config_file
    expected_sections = [
        "global_params",
        "experiment",
        "data",
        "models",
    ]
    if "single_run" in config_name:
        expected_sections.append("run_params")
    elif "hyperopt" in config_name:
        expected_sections.extend(["hyperopt_search_space", "hyperopt_settings"])

    for section in expected_sections:
        assert section in config_data, f"Missing top-level section '{section}' in {config_name}"


def test_global_params_structure_and_types(config_file):
    """Tests the structure and data types within the global_params section."""
    config_name, config_data = config_file
    global_params = config_data.get("global_params", {})
    assert "verbose" in global_params, f"Missing 'verbose' in global_params in {config_name}"
    assert isinstance(global_params["verbose"], int), f"'verbose' should be an integer in {config_name}."
    assert "error_raise" in global_params, f"Missing 'error_raise' in global_params in {config_name}"
    assert isinstance(global_params["error_raise"], bool), f"'error_raise' should be a boolean in {config_name}."


def test_run_params_structure(config_file):
    """Tests for the presence of key parameters in the run_params section."""
    config_name, config_data = config_file
    if "single_run" in config_name:
        run_params = config_data.get("run_params", {})
        assert "outcome_var_n" in run_params, "Missing 'outcome_var_n' in run_params"
        assert "scale" in run_params, "Missing 'scale' in run_params"
        assert "feature_n" in run_params, "Missing 'feature_n' in run_params"
        assert "data" in run_params, "Missing 'data' sub-dictionary in run_params"
        assert isinstance(run_params["data"], dict), "'data' in run_params should be a dictionary."


def test_hyperopt_search_structure(config_file):
    """Tests for the presence of key parameters in the hyperopt_search section."""
    config_name, config_data = config_file
    if "hyperopt" in config_name:
        hyperopt_search = config_data.get("hyperopt_search_space", {})
        assert "resample" in hyperopt_search, "Missing 'resample' in hyperopt_search_space"
        assert "data" in hyperopt_search, "Missing 'data' sub-dictionary in hyperopt_search_space"
        hyperopt_settings = config_data.get("hyperopt_settings", {})
        assert "max_evals" in hyperopt_settings, "Missing 'max_evals' in hyperopt_settings"