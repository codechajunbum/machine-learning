# ML Bayesian Network

Bayesian classifier and Gaussian Naive Bayes from scratch with Laplace smoothing.

## Features
- Categorical Bayesian classifier with Laplace smoothing
- Gaussian Naive Bayes with variance smoothing
- Log-probability computation for numerical stability
- scikit-learn compatible interface

## Usage
```python
from src.bayesian import GaussianNaiveBayes
model = GaussianNaiveBayes(var_smoothing=1e-9)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```
