import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 subsample=0.8, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.trees = []
        self.init_pred = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _log_loss_gradient(self, y, pred):
        p = self._sigmoid(pred)
        return y - p

    def fit(self, X, y):
        n = X.shape[0]
        p = np.mean(y)
        self.init_pred = np.log(p / (1 - p + 1e-10))
        F = np.full(n, self.init_pred)

        for _ in range(self.n_estimators):
            residuals = self._log_loss_gradient(y, F)
            idx = np.random.choice(n, int(n * self.subsample), replace=False)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X[idx], residuals[idx])
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        prob = self._sigmoid(F)
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class XGBoostLite:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 lambda_=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.gamma = gamma
        self.trees = []
        self.init_pred = 0.5

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X, y):
        F = np.full(len(y), np.log(self.init_pred / (1 - self.init_pred)))
        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            grad = p - y
            hess = p * (1 - p) + self.lambda_
            pseudo_resp = -grad / (hess + 1e-10)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, pseudo_resp, sample_weight=hess)
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        F = np.full(X.shape[0], np.log(self.init_pred / (1 - self.init_pred)))
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return (self._sigmoid(F) >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
