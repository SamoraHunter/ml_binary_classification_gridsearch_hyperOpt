# from sklearn.neighbors import KNeighborsClassifier

from simbsig.neighbors import KNeighborsClassifier
from sklearn import metrics


class KNNWrapper:
    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        device="gpu",
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.device = device

        # self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p, metric=self.metric, metric_params=self.metric_params)

    def fit(self, X, y):
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            metric=self.metric,
            metric_params=self.metric_params,
            device=self.device,
        )

        self.model.fit(X, y)
        return self

    def get_params(self, deep=False):
        return {
            "device": self.device,
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "metric": self.metric,
            "metric_params": self.metric_params,
            "n_neighbors": self.n_neighbors,
        }

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        # return 1
        # return self.model.score(y, self.predict(X))
        y_pred = self.predict(X)
        return metrics.accuracy_score(y, y_pred)
        # return self.model.get_acc(self.predict(X), y)

    #     def decision_function(self, X, y):
    #         return self.model.predict_proba(X)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


#     def set_params(self):
#         self.device = 'gpu'
