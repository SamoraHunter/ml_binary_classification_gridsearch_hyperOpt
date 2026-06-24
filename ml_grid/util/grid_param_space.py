"""Defines the Grid class for creating a hyperparameter search space."""

import itertools as it
import logging
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
        self.logger = logging.getLogger("ml_grid")

        if sample_n is None:
            self.sample_n = 1000
        else:
            self.sample_n = sample_n

        self.logger.info(f"Feature space slice sample_n: {self.sample_n}")

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
            "feature_selection_method": ["anova", "markov_blanket"],
            "use_embedding": [True, False],
            "embedding_method": ["pca", "svd"],
            "embedding_dim": [32, 64, 128],
            "scale_features_before_embedding": [True],
            "cache_embeddings": [False],
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

    def _c_prod(self, d: Union[Dict, List]) -> Generator[Dict, None, None]:
        """Recursively generates the Cartesian product of a nested dictionary.

        Args:
            d (Union[Dict, List]): The dictionary or list of settings.

        Yields:
            Generator[Dict, None, None]: A generator of dictionaries, each
            representing a unique combination of settings.
        """
        if isinstance(d, list):
            for i in d:
                yield from ([i] if not isinstance(i, (dict, list)) else self._c_prod(i))
        else:
            for i in it.product(*map(self._c_prod, d.values())):
                yield dict(zip(d.keys(), i))

    def _generate_full_grid(self) -> List[Dict]:
        """Generates the full Cartesian product of grid settings."""
        return list(self._c_prod(self.grid))

    @property
    def settings_list(self) -> List[Dict]:
        """Lazily generate and cache the settings list."""
        if not hasattr(self, "_settings_list"):
            full_grid = self._generate_full_grid()
            random.shuffle(full_grid)

            sample_size = min(self.sample_n, len(full_grid))
            if self.sample_n > len(full_grid):
                self.logger.warning(
                    f"sample_n ({self.sample_n}) is larger than the number of settings ({len(full_grid)}). Using all settings."
                )

            self._settings_list = random.sample(full_grid, sample_size)
        return self._settings_list

    @property
    def settings_list_iterator(self) -> Generator[Dict, None, None]:
        """Returns an iterator over settings list."""
        return iter(self.settings_list)
