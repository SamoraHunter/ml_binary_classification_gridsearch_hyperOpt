from numpy import absolute, mean, std
from sklearn import metrics
from sklearn.metrics import (
    classification_report,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    roc_auc_score,
)


class debug_print_statements_class:

    def __init__(self, scores):

        self.scores = scores

    def debug_print_scores(self):
        """Print mean and standard deviation of scores in a grid search
        
        Parameters
        ----------
        scores : dict
            Dictionary containing the scores for each parameter combination
        """
        try:
            print(
                "Mean MAE: %.3f (%.3f)"
                % (
                    absolute(mean(self.scores["test_f1"])),
                    std(self.scores["test_f1"]),
                )
            )
        except Exception as e:
            print("Error printing Mean MAE:", e)
        
        try:
            print(
                "Mean ROC AUC: %.3f (%.3f)"
                % (
                    absolute(mean(self.scores["test_roc_auc"])),
                    std(self.scores["test_roc_auc"]),
                )
            )
        except Exception as e:
            print("Error printing Mean ROC AUC:", e)
        
        try:
            print(
                "Mean accuracy: %.3f (%.3f)"
                % (absolute(mean(self.scores["test_accuracy"])), std(self.scores["test_accuracy"]))
            )
        except Exception as e:
            print("Error printing Mean accuracy:", e)
        
        try:
            print(
                "Mean fit time: %.3f (%.3f)"
                % (absolute(mean(self.scores["fit_time"])), std(self.scores["fit_time"]))
            )
        except Exception as e:
            print("Error printing Mean fit time:", e)
        
        try:
            print(
                "Mean score time: %.3f (%.3f)"
                % (absolute(mean(self.scores["score_time"])), std(self.scores["score_time"]))
            )
        except Exception as e:
            print("Error printing Mean score time:", e)
        
        try:
            print(
                "---------------------------------------------------------------------------------------------------"
            )
        except Exception as e:
            print("Error printing Separator:", e)

