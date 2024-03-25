"""Import parameter space constants"""

import numpy as np

# print("imported param_space")


class ParamSpace:

    # print("param space called")

    def __init__(self, size):

        # print("param space init called")

        self.param_dict = None

        if size == "medium":

            nstep = 3
            self.param_dict = {
                "log_epoch": np.floor(np.logspace(0, 1.5, 3)).astype(int),
                "log_small": np.logspace(-1, -5, 3),
                "bool_param": [True, False],
                "log_large": np.logspace(0, 2, 3).astype(int),
                "log_large_long": np.floor(np.logspace(0, 3.1, 5)).astype(int),
                "log_med_long": np.floor(np.logspace(0, 1.5, 5)).astype(int),
                "log_med": np.floor(np.logspace(0, 1.5, 3)).astype(int),
                "log_zero_one": np.logspace(0.0, 1.0, nstep) / 10,
                "lin_zero_one": np.linspace(0.0, 1.0, nstep) / 10,
            }

        if size == "xsmall":

            nstep = 2
            self.param_dict = {
                "log_epoch": np.floor(np.logspace(-1, 1, 3)).astype(int),
                "log_small": np.logspace(-1, -5, 2),
                "bool_param": [True, False],
                "log_large": np.logspace(0, 2, 2).astype(int),
                "log_large_long": np.floor(np.logspace(0, 3.1, 2)).astype(int),
                "log_med_long": np.floor(np.logspace(0, 1.5, 2)).astype(int),
                "log_med": np.floor(np.logspace(0, 1.5, 2)).astype(int),
                "log_zero_one": np.logspace(0.0, 1.0, nstep) / 10,
                "lin_zero_one": np.linspace(0.0, 1.0, nstep) / 10,
            }

        if size == "xwide":

            nstep = 2
            self.param_dict = {
                "log_epoch": np.floor(np.logspace(-2, 2, 3)).astype(int),
                "log_small": np.logspace(-1, -5, 2),
                "bool_param": [True, False],
                "log_large": np.logspace(0, 2, 2).astype(int),
                "log_large_long": np.floor(np.logspace(0, 3.1, 2)).astype(int),
                "log_med_long": np.floor(np.logspace(0, 1.5, 2)).astype(int),
                "log_med": np.floor(np.logspace(0, 1.5, 2)).astype(int),
                "log_zero_one": np.logspace(0.0, 1.0, nstep) / 10,
                "lin_zero_one": np.linspace(0.0, 1.0, nstep) / 10,
            }
