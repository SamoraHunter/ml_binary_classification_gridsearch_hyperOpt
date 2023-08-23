
from numpy import absolute, mean, std
from sklearn import metrics
from sklearn.metrics import (classification_report, f1_score, make_scorer,
                             matthews_corrcoef, roc_auc_score)


class debug_print_statements_class():
    
    def __init__(self, scores):
        
        self.scores = scores
        
        
    
    def debug_print_scores(scores):
        
        print(
            "Mean MAE: %.3f (%.3f)"
            % (
                absolute(mean(scores["test_f1"])),
                std(scores["test_f1"]),
            )
        )
        print(
            "Mean ROC AUC: %.3f (%.3f)"
            % (
                absolute(mean(scores["test_roc_auc"])),
                std(scores["test_roc_auc"]),
            )
        )
        print(
            "Mean accuracy: %.3f (%.3f)"
            % (absolute(mean(scores["test_accuracy"])), std(scores["test_accuracy"]))
        )
        print(
            "Mean fit time: %.3f (%.3f)"
            % (absolute(mean(scores["fit_time"])), std(scores["fit_time"]))
        )
        print(
            "Mean score time: %.3f (%.3f)"
            % (absolute(mean(scores["score_time"])), std(scores["score_time"]))
        )
        print(
            "---------------------------------------------------------------------------------------------------"
        )