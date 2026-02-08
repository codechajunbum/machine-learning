import numpy as np
from collections import defaultdict


class BayesianClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.cond_probs = {}
        self.classes = None
        self.n_features = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        n = len(y)

        for cls in self.classes:
            mask = y == cls
            self.class_priors[cls] = np.sum(mask) / n
            X_cls = X[mask]
            self.cond_probs[cls] = {}
            for feat in range(self.n_features):
                values = np.unique(X[:, feat])
                total = len(X_cls) + self.alpha * len(values)
                self.cond_probs[cls][feat] = {
                    v: (np.sum(X_cls[:, feat] == v) + self.alpha) / total
                    for v in values
                }
        return self

    def predict_proba(self, X):
        probs = np.zeros((len(X), len(self.classes)))
        for i, x in enumerate(X):
            for j, cls in enumerate(self.classes):
                log_prob = np.log(self.class_priors[cls])
                for feat in range(self.n_features):
                    prob = self.cond_probs[cls][feat].get(x[feat], self.alpha / (len(X) + self.alpha))
                    log_prob += np.log(prob + 1e-10)
                probs[i, j] = log_prob
        probs -= probs.max(axis=1, keepdims=True)
        probs = np.exp(probs)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes = None
        self.priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.priors[cls] = len(X_cls) / len(y)
            self.means[cls] = X_cls.mean(axis=0)
            self.vars[cls] = X_cls.var(axis=0) + self.var_smoothing
        return self

    def _log_likelihood(self, x, cls):
        mean, var = self.means[cls], self.vars[cls]
        return -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)

    def predict(self, X):
        preds = []
        for x in X:
            scores = {cls: np.log(self.priors[cls]) + self._log_likelihood(x, cls)
                      for cls in self.classes}
            preds.append(max(scores, key=scores.get))
        return np.array(preds)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
