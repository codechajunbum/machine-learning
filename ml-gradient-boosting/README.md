# ML Gradient Boosting

Gradient Boosting Machine (GBM) and XGBoost-lite classifier from scratch using decision tree weak learners.

## Features
- GBM with log-loss gradient and subsampling
- XGBoost-lite with second-order gradient (Hessian)
- Configurable learning rate, depth, and regularization
- scikit-learn compatible interface

## Usage
```python
from src.gbm import GradientBoostingClassifier, XGBoostLite
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```
