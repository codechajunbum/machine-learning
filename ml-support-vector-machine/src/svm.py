import numpy as np


class SVM:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3, tol=1e-3, max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        self.alphas = None
        self.b = 0
        self.support_vectors = None
        self.support_labels = None
        self.support_alphas = None

    def _kernel(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            gamma = (1 / (X1.shape[1] * X1.var())
                     if self.gamma == 'scale' else self.gamma)
            dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                    np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
            return np.exp(-gamma * dists)
        elif self.kernel == 'poly':
            return (X1 @ X2.T + 1) ** self.degree

    def fit(self, X, y):
        n = X.shape[0]
        y = np.where(y <= 0, -1, 1).astype(float)
        K = self._kernel(X, X)
        self.alphas = np.zeros(n)
        self.b = 0

        for _ in range(self.max_iter):
            alpha_prev = self.alphas.copy()
            for i in range(n):
                decision = np.sum(self.alphas * y * K[:, i]) + self.b
                error_i = decision - y[i]
                if (y[i] * error_i < -self.tol and self.alphas[i] < self.C) or \
                   (y[i] * error_i > self.tol and self.alphas[i] > 0):
                    j = np.random.choice([x for x in range(n) if x != i])
                    error_j = np.sum(self.alphas * y * K[:, j]) + self.b - y[j]
                    ai_old, aj_old = self.alphas[i], self.alphas[j]

                    L = max(0, aj_old - ai_old) if y[i] != y[j] else max(0, ai_old + aj_old - self.C)
                    H = min(self.C, self.C + aj_old - ai_old) if y[i] != y[j] else min(self.C, ai_old + aj_old)
                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alphas[j] -= y[j] * (error_i - error_j) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    if abs(self.alphas[j] - aj_old) < 1e-5:
                        continue

                    self.alphas[i] += y[i] * y[j] * (aj_old - self.alphas[j])
                    b1 = self.b - error_i - y[i] * (self.alphas[i]-ai_old)*K[i,i] - y[j]*(self.alphas[j]-aj_old)*K[i,j]
                    b2 = self.b - error_j - y[i] * (self.alphas[i]-ai_old)*K[i,j] - y[j]*(self.alphas[j]-aj_old)*K[j,j]
                    self.b = (b1 + b2) / 2

            if np.max(np.abs(self.alphas - alpha_prev)) < self.tol:
                break

        sv_idx = self.alphas > 1e-5
        self.support_vectors = X[sv_idx]
        self.support_labels = y[sv_idx]
        self.support_alphas = self.alphas[sv_idx]
        return self

    def predict(self, X):
        K = self._kernel(X, self.support_vectors)
        decision = K @ (self.support_alphas * self.support_labels) + self.b
        return np.where(decision >= 0, 1, 0)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
