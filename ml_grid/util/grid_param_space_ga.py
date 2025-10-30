"""Defines the Grid class for creating a hyperparameter search space for GA."""

import itertools as it
import logging
import random
from typing import Dict, Generator, List, Optional, Union

from ml_grid.util.global_params import global_parameters


class Grid:
    """Generates and manages a grid of hyperparameter settings for GA experiments."""

    global_params: global_parameters
    """A reference to the global parameters singleton instance."""

    verbose: int
    """The verbosity level, inherited from global parameters."""

    sample_n: int
    """The number of random settings to sample from the full grid."""

    grid: Dict[str, Union[List, Dict]]
    """
    The dictionary defining the hyperparameter search space for the GA.

    Keys represent different aspects of the experiment:
    - **weighted**: The weighting strategy for the ensemble.
    - **use_stored_base_learners**: Whether to use pre-trained base learners.
    - **store_base_learners**: Whether to save the base learners after training.
    - **resample**: The resampling strategy to handle class imbalance.
    - **scale**: Whether to apply standard scaling to features.
    - **n_features**: The number of features to use (currently 'all').
    - **param_space_size**: The size of the hyperparameter space for base learners.
    - **n_unique_out**: A parameter for future use.
    - **outcome_var_n**: The index of the outcome variable to use.
    - **div_p**: A parameter for future use.
    - **percent_missing**: The threshold for dropping columns with missing values.
    - **corr**: The threshold for dropping highly correlated features.
    - **cxpb**: The crossover probability for the genetic algorithm.
    - **mutpb**: The mutation probability for the genetic algorithm.
    - **indpb**: The independent probability for each attribute to be mutated.
    - **t_size**: The tournament size for selection in the genetic algorithm.
    - **data**: A nested dictionary specifying which feature categories to include.
    """

    settings_list: List[Dict]
    """
    A list of hyperparameter combinations sampled from the `grid`. Each element
    is a dictionary representing one complete experimental configuration.
    """

    settings_list_iterator: it.chain
    """An iterator over the `settings_list`."""

    def __init__(self, sample_n: Optional[int] = 1000):
        """Initializes the Grid object for Genetic Algorithms.

        This class creates a comprehensive grid of settings by taking the
        Cartesian product of all specified hyperparameters. It then randomly
        samples a specified number of these settings to create a manageable
        list for experimentation.

        Args:
            sample_n (Optional[int], optional): The number of random settings to
                sample from the full grid. Defaults to 1000.
        """

        self.global_params = global_parameters

        self.verbose = self.global_params.verbose
        self.logger = logging.getLogger("ml_grid")

        if sample_n is None:
            self.sample_n = 1000
        else:
            self.sample_n = sample_n

        if self.verbose >= 1:
            self.logger.info(f"Feature space slice sample_n {self.sample_n}")

        # Default grid
        # User can update grid dictionary on the object
        self.grid = {
            "weighted": ["ann", "de", "unweighted"],
            "use_stored_base_learners": [False],
            "store_base_learners": [False],
            "resample": ["undersample", None],
            "scale": [True],
            "n_features": ["all"],
            "param_space_size": ["medium"],
            "n_unique_out": [10],
            "outcome_var_n": ["1"],
            "div_p": [0],
            "percent_missing": [99.9, 95, 90],  # n/100 ex 95 for 95%
            "corr": [0.8, 0.5],
            "cxpb": [0.5, 0.75, 0.25],
            "mutpb": [0.2, 0.4, 0.8],
            "indpb": [0.025, 0.05, 0.075],
            "t_size": [3, 6, 9],
            "data": [
                {
                    "age": [True],
                    "sex": [True],
                    "bmi": [True],
                    "ethnicity": [True],
                    "bloods": [True, False],
                    "diagnostic_order": [True, False],
                    "drug_order": [True, False],
                    "annotation_n": [True, False],
                    "meta_sp_annotation_n": [True, False],
                    "annotation_mrc_n": [True, False],
                    "meta_sp_annotation_mrc_n": [True, False],
                    "core_02": [False],
                    "bed": [False],
                    "vte_status": [True],
                    "hosp_site": [True],
                    "core_resus": [False],
                    "news": [False],
                }
            ],
        }

        def c_prod(d: Union[Dict, List]) -> Generator[Dict, None, None]:
            """Recursively generates the Cartesian product of a nested dictionary.

            Args:
                d (Union[Dict, List]): The dictionary or list of settings.

            Yields:
                Generator[Dict, None, None]: A generator of dictionaries, each
                representing a unique combination of settings.
            """
            if isinstance(d, list):
                for i in d:
                    yield from ([i] if not isinstance(i, (dict, list)) else c_prod(i))
            else:
                for i in it.product(*map(c_prod, d.values())):
                    yield dict(zip(d.keys(), i))

        self.settings_list = list(c_prod(self.grid))
        full_settings_size = len(self.settings_list)
        self.logger.info(f"Full settings_list size: {full_settings_size}")

        random.shuffle(self.settings_list)

        # Ensure sample_n is not greater than the number of available settings
        sample_size = min(self.sample_n, full_settings_size)
        if self.sample_n > full_settings_size:
            self.logger.warning(
                f"sample_n ({self.sample_n}) is larger than the number of settings ({full_settings_size}). Using all settings."
            )

        self.settings_list = random.sample(self.settings_list, sample_size)

        self.settings_list_iterator = iter(self.settings_list)
