import logging
import tempfile
from pathlib import Path

import pytest

from ml_grid.pipeline.data import pipe

# --- Essential imports from the ml_grid project ---
from ml_grid.pipeline.main import run
from ml_grid.util.create_experiment_directory import create_experiment_directory
from ml_grid.util.global_params import global_parameters
from ml_grid.util.impute_data_for_pipe import mean_impute_dataframe
from ml_grid.util.synthetic_data_generator import generate_synthetic_data

# Suppress verbose logging from the application during tests
logging.basicConfig(level=logging.CRITICAL)

# List of all H2O model class names to be tested individually
H2O_MODELS_TO_TEST = [
    "H2O_GBM_class",
    "H2O_DRF_class",
    "H2O_DeepLearning_class",
    "H2O_GLM_class",
    "H2O_NaiveBayes_class",
    "H2O_RuleFit_class",
    "H2O_XGBoost_class",
    "H2O_GAM_class",
    # 'H2O_StackedEnsemble_class' is excluded because it requires pre-trained
    # base models, which is outside the scope of this test.
]


@pytest.fixture(scope="module")
def pipeline_config():
    """
    A pytest fixture to set up a minimal pipeline configuration for testing.
    This runs once per module, creating a shared dataset and config structure.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Generate and save a synthetic dataset
        synthetic_df, important_feature_map = generate_synthetic_data(
            n_rows=600, n_features=40, n_outcome_vars=1
        )
        outcome_columns = list(important_feature_map.keys())
        imputed_df = mean_impute_dataframe(data=synthetic_df, y_vars=outcome_columns)
        data_path = temp_path / "synthetic_test_data.csv"
        imputed_df.to_csv(data_path, index=False)

        # 2. Configure global parameters for a fast, non-interactive test run
        global_parameters.test_mode = True
        global_parameters.verbose = 0
        global_parameters.h2o_show_progress = False
        global_parameters.n_iter = 2
        global_parameters.error_raise = True

        # 3. Create a base config structure
        experiment_base_dir = temp_path / "experiments"
        experiment_base_dir.mkdir()

        base_config = {
            "experiment": {
                "experiments_base_dir": str(experiment_base_dir),
                "additional_naming": "integration_test",
            },
            "global_params": {"verbose": 0, "error_raise": True},
            "data": {
                "file_path": str(data_path),
                "outcome_var_override": outcome_columns[0],
                "drop_term_list": [],
            },
            "run_params": {"param_space_size": "xsmall"},
        }

        # Yield the necessary components to the test functions
        yield {
            "base_config": base_config,
            "base_project_dir": str(temp_path),
            "experiment_base_dir": experiment_base_dir,
            "data_path": str(data_path),
            "outcome_var": outcome_columns[0],
        }
        # Cleanup is handled by TemporaryDirectory context manager


@pytest.mark.parametrize("model_to_test", H2O_MODELS_TO_TEST)
def test_h2o_model_execution(pipeline_config, model_to_test, h2o_session_fixture):
    """
    Tests that the main `run.execute()` completes successfully for each H2O model.
    This validates the data flow through fit, predict, and score for each one.
    """
    # 1. Create a specific config for this model run
    # The h2o_session_fixture ensures the H2O cluster is already running.
    test_config = pipeline_config["base_config"].copy()
    # Create a model dictionary that includes all models but only enables the one being tested.
    test_config["models"] = {  # pylint: disable=duplicate-key
        model: (model == model_to_test) for model in H2O_MODELS_TO_TEST
    }

    # Construct the local_param_dict with the expected nested structure
    local_params = test_config["run_params"].copy()
    local_params["data"] = test_config["data"]

    # 2. Instantiate the real `pipe` object for this specific test
    ml_grid_object = pipe(
        file_name=pipeline_config["data_path"],
        drop_term_list=[],
        local_param_dict=local_params,
        base_project_dir=pipeline_config["base_project_dir"],
        experiment_dir=create_experiment_directory(
            pipeline_config["experiment_base_dir"], f"test_{model_to_test}"
        ),
        test_sample_n=0,
        param_space_index=0,
        model_class_dict=test_config["models"],  # Pass the single-model config
        outcome_var_override=pipeline_config["outcome_var"],
    )

    # 3. Define the local parameters for the run
    local_param_dict = local_params

    # 4. Execute the pipeline and assert success
    try:
        run_instance = run(local_param_dict, ml_grid_object=ml_grid_object)
        # The following lines are commented out as they are not directly relevant
        # to the user's issue and might cause confusion. The primary goal is to
        # ensure the run completes without exceptions.
        # model_errors, highest_score = run_instance.execute()
        # assert len(model_errors) == 0, (
        #     f"The pipeline for {model_to_test} should execute without any model errors."
        # )
        model_errors, highest_score = run_instance.execute()
        assert (
            len(model_errors) == 0
        ), f"The pipeline for {model_to_test} should execute without any model errors."
    except Exception as e:
        pytest.fail(
            f"The `run.execute()` method for {model_to_test} raised an "
            f"unexpected exception: {e}"
        )
