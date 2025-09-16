from typing import Dict

import numpy as np


class debug_print_statements_class:
    """A class for printing debug statements related to model scores."""

    def __init__(self, scores: Dict[str, np.ndarray]):
        """Initializes the debug_print_statements_class.

        Args:
            scores (Dict[str, np.ndarray]): A dictionary of scores from a
                scikit-learn cross-validation run. Expected keys include
                'test_f1', 'test_roc_auc', 'test_accuracy', 'fit_time',
                and 'score_time'.
        """
        self.scores = scores

    def debug_print_scores(self) -> None:
        """Prints the mean and standard deviation of various scores.

        This method iterates through a predefined set of score keys,
        calculates the mean and standard deviation for each, and prints
        the results to the console.
        """
        # The original code printed "Mean MAE" but used the "test_f1" score.
        # Changed the label to "Mean F1" for clarity.
        score_keys = {
            "test_f1": "Mean F1",
            "test_roc_auc": "Mean ROC AUC",
            "test_accuracy": "Mean accuracy",
            "fit_time": "Mean fit time",
            "score_time": "Mean score time",
        }

        for key, label in score_keys.items():
            try:
                if key in self.scores:
                    score_mean = np.absolute(np.mean(self.scores[key]))
                    score_std = np.std(self.scores[key])
                    print(f"{label}: {score_mean:.3f} ({score_std:.3f})")
            except Exception as e:
                print(f"Error printing {label}: {e}")

        print("-" * 80)
