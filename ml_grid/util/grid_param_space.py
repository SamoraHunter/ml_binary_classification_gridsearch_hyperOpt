"""Defines the Grid class for creating a hyperparameter search space."""

import itertools as it
import random
from typing import Dict, Generator, List, Optional, Union

from ml_grid.util.global_params import global_parameters


class Grid:
    """Generates and manages a grid of hyperparameter settings for experiments."""

    def __init__(self, sample_n: Optional[int] = 1000):
        """Initializes the Grid object.

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

        if sample_n is None:
            self.sample_n = 1000
        else:
            self.sample_n = sample_n

        if self.verbose >= 1:
            print(f"Feature space slice sample_n {self.sample_n}")

        # Default grid
        # User can update grid dictionary on the object
        self.grid = {
            "resample": ["undersample", "oversample", None],
            "scale": [True, False],
            "feature_n": [100, 95, 75, 50, 25, 5],
            "param_space_size": ["medium", "xsmall"],
            "n_unique_out": [10],
            "outcome_var_n": ["1"],
            "percent_missing": [99, 95, 80],  # n/100 ex 95 for 95% # 99.99, 99.5, 9
            "corr": [0.98, 0.85, 0.5, 0.25],
            "feature_selection_method":['anova', 'markov_blanket'],
            "data": [
                {
                    "age": [True, False],
                    "sex": [True, False],
                    "bmi": [True],
                    "ethnicity": [True, False],
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
                    "date_time_stamp": [False],
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
        print(f"Full settings_list size: {full_settings_size}")

        random.shuffle(self.settings_list)

        # Ensure sample_n is not greater than the number of available settings
        sample_size = min(self.sample_n, full_settings_size)
        if self.sample_n > full_settings_size and self.verbose >= 1:
            print(f"Warning: sample_n ({self.sample_n}) is larger than the number of settings ({full_settings_size}). Using all settings.")

        self.settings_list = random.sample(self.settings_list, sample_size)

        self.settings_list_iterator = iter(self.settings_list)

        # This is likely not properly functioning. Does not return iteration, instead reinitiates.
        # Don't need to subsample, can just generate n number of random choices from grid space.
        # function can just return random choice from grid space, terminate at the other end once limit reached.
