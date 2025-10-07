import pathlib
import time
import traceback
import numpy as np
import pandas as pd
from ml_grid.util.global_params import global_parameters
from sklearn import metrics
from sklearn.metrics import *
import pickle
import os
import warnings
from typing import Any, Dict, List

# from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def _get_score_log_columns(metric_list: List[str]) -> List[str]:
    """Generates the list of column names for the score log file.

    Args:
        metric_list (List[str]): A list of metric names to include.

    Returns:
        List[str]: A comprehensive list of column names for the log.
    """
    column_list: List[str] = [
        "algorithm_implementation", "parameter_sample", "method_name", "nb_size",
        "f_list", "auc", "mcc", "f1", "precision", "recall", "accuracy",
        "resample", "scale", "n_features", "param_space_size", "n_unique_out",
        "outcome_var_n", "percent_missing", "corr", "age", "sex", "bmi",
        "ethnicity", "bloods", "diagnostic_order", "drug_order", "annotation_n",
        "meta_sp_annotation_n", "meta_sp_annotation_mrc_n", "annotation_mrc_n", 
        "use_embedding", "embedding_method", "embedding_dim", 
        "scale_features_before_embedding", "cache_embeddings",
        "core_02", "bed", "vte_status", "hosp_site", "core_resus", "news",
        "date_time_stamp", "X_train_size", "X_test_orig_size", "X_test_size",
        "run_time", "n_fits", "t_fits", "i", "outcome_variable", 'failed'
    ]

    metric_names: List[str] = []
    for metric in metric_list:
        metric_names.append(f"{metric}_m")
        metric_names.append(f"{metric}_std")

    column_list.extend(metric_names)
    return column_list

class project_score_save_class:
    """Handles the creation and updating of the project's score log file."""

    def __init__(self, base_project_dir: str):
        """Initializes the score logger and creates the log file with headers.

        Args:
            base_project_dir (str): The root directory for the project where
                the log file will be saved.
        """

        warnings.filterwarnings("ignore")

        warnings.filterwarnings("ignore", category=FutureWarning)

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        warnings.filterwarnings("ignore", category=UserWarning)

        self.global_params = global_parameters

        self.metric_list = self.global_params.metric_list

        self.error_raise = self.global_params.error_raise

        self.column_list = _get_score_log_columns(list(self.metric_list.keys()))

        # column_list = column_list +['BL_' + str(x) for x in range(0, 64)]

        df = pd.DataFrame(data=None, columns=self.column_list)

        df.to_csv(
            os.path.join(base_project_dir, "final_grid_score_log.csv"),
            mode="w",
            header=True,
            index=False,
        )

    @staticmethod
    def update_score_log(
        
        ml_grid_object: Any,
        scores: Dict[str, np.ndarray],
        best_pred_orig: np.ndarray,
        current_algorithm: Any,
        method_name: str,
        pg: int,
        start: float,
        n_iter_v: int,
        failed: bool,
    ):
        """Updates the score log with the results of a single experiment run.

        Args:
            ml_grid_object (Any): The main pipeline object containing all data
                and parameters for the current iteration.
            scores (Dict[str, np.ndarray]): A dictionary of scores from
                cross-validation.
            best_pred_orig (np.ndarray): Predictions from the best estimator on
                the original test set.
            current_algorithm (Any): The best estimator instance from the search.
            method_name (str): The name of the algorithm method.
            pg (int): The size of the parameter grid.
            start (float): The start time of the run (from `time.time()`).
            n_iter_v (int): The number of iterations performed in the search.
            failed (bool): A flag indicating if the run failed.
        """

        global_params = global_parameters

        ml_grid_object_iter = ml_grid_object

        X_train = ml_grid_object_iter.X_train

        y_train = ml_grid_object_iter.y_train

        X_test = ml_grid_object_iter.X_test

        y_test = ml_grid_object_iter.y_test

        X_test_orig = ml_grid_object_iter.X_test_orig

        y_test_orig = ml_grid_object_iter.y_test_orig

        param_space_index = ml_grid_object.param_space_index
        
        bayessearch = global_params.bayessearch

        store_models = global_params.store_models
        # n_iter_v = np.nan ##????????????

        try:
            print("Writing grid permutation to log")
            # write line to best grid scores---------------------
            column_list = _get_score_log_columns(list(global_params.metric_list.keys()))
            line = pd.DataFrame(data=None, columns=column_list)

            # best_pred_orig = grid.best_estimator_.predict(X_test_orig)
            try:
                auc = metrics.roc_auc_score(y_test, best_pred_orig)
            except Exception as e:
                if ml_grid_object.verbose >= 1:
                    print(best_pred_orig)
                    print(e)
                auc = np.nan

            mcc = matthews_corrcoef(y_test, best_pred_orig)
            f1 = f1_score(y_test, best_pred_orig, average="binary")
            precision = precision_score(y_test, best_pred_orig, average="binary")
            recall = recall_score(y_test, best_pred_orig, average="binary")
            accuracy = accuracy_score(y_test, best_pred_orig)

            # get info from current settings iter...local_param_dict ml_grid_object
            for key in ml_grid_object.local_param_dict:
                # print(key)
                if key != "data":
                    if key in column_list:
                        line[key] = [ml_grid_object.local_param_dict.get(key)]
                else:
                    for key_1 in ml_grid_object.local_param_dict.get("data"):
                        # print(key_1)
                        if key_1 in column_list:
                            line[key_1] = [
                                ml_grid_object.local_param_dict.get("data").get(key_1)
                            ]

            current_f = ml_grid_object.final_column_list
            # current_f = list(self.X_test.columns)
            current_f_vector = []
            f_list = []
            for elem in ml_grid_object.orignal_feature_names:
                if elem in current_f:
                    current_f_vector.append(1)
                else:
                    current_f_vector.append(0)
            # f_list.append(np.array(current_f_vector))
            f_list.append(current_f_vector)

            line["algorithm_implementation"] = [current_algorithm]
            line["parameter_sample"] = [current_algorithm]
            line["method_name"] = [method_name]
            line["nb_size"] = [sum(np.array(current_f_vector))]
            line["n_features"] = [len(current_f_vector)]
            line["f_list"] = [f_list]

            line["auc"] = [auc]
            line["mcc"] = [mcc]
            line["f1"] = [f1]
            line["precision"] = [precision]
            line["recall"] = [recall]
            line["accuracy"] = [accuracy]

            line["X_train_size"] = [len(X_train)]
            line["X_test_orig_size"] = [len(X_test_orig)]
            line["X_test_size"] = [len(X_test)]

            end = time.time()
            
            print("Debug scores:")
            print(scores)
            
        
                

            line["run_time"] = int((end - start) / 60)
            line["t_fits"] = pg
            line["n_fits"] = n_iter_v
            line["i"] = param_space_index  # 0 # should be index of the iterator
            line['outcome_variable'] = ml_grid_object_iter.outcome_variable
            line['failed'] = failed
            
            if bayessearch:
                try:
                    line["fit_time_m"] = np.array([scores["fit_time"]]).mean()
                    line["fit_time_std"] = np.array([scores["fit_time"]]).std()
                    
                    line["score_time_m"] = np.array(scores["score_time"]).mean()
                    line["score_time_std"] = np.array(scores["score_time"]).std()
                    
                    for metric in global_params.metric_list:
                        line[f"{metric}_m"] = np.array(scores[f"test_{metric}"]).mean()
                        line[f"{metric}_std"] = np.array(scores[f"test_{metric}"]).std()
                    
                except Exception as e:
                    print(e)
                    print(scores)
                    raise e
            else:
                line["fit_time_m"] = scores["fit_time"].mean() #deprecated for bayes
                line["fit_time_std"] = scores["fit_time"].std()
                line["score_time_m"] = scores["score_time"].mean()
                line["score_time_std"] = scores["score_time"].std()
                
                for metric in global_params.metric_list:
                    line[f"{metric}_m"] = scores[f"test_{metric}"].mean()
                    line[f"{metric}_std"] = scores[f"test_{metric}"].std()

            
            
            

            print(line)

            # line['outcome_var'] = y_test.name

            # line['nb_val'] = [nb_val]
            # line['pop_val'] = [pop_val]
            # line['g_val'] = [g_val]
            # line['g'] = [g]

            line[column_list].to_csv(
                os.path.join(ml_grid_object.base_project_dir, "final_grid_score_log.csv"),
                mode="a",
                header=False,
                index=True,
            )
            if store_models:
                if "keras" not in method_name.lower():
                    #print("SAVING MODEL!")
                    models_dir = pathlib.Path(os.path.join(ml_grid_object.base_project_dir, "models"))
                    models_dir.mkdir(parents=True, exist_ok=True)

                    model_path = os.path.join(ml_grid_object.base_project_dir, "models", f"{str(param_space_index)}.pkl")
                    try:
                        # save pickled model
                        with open(model_path, 'wb') as f:
                            pickle.dump(current_algorithm, f)
                    except Exception as e:
                        print(e)
                        raise e
            # ---------------------------
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("Failed to upgrade grid entry")
            if global_params.error_raise:
                input()
