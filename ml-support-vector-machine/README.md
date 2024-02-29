# ML Support Vector Machine

SVM classifier from scratch using SMO (Sequential Minimal Optimization) with RBF, Linear, and Polynomial kernels.

## Features
- SMO optimization algorithm
- RBF, Linear, Polynomial kernels
- Support vector extraction
- Configurable C regularization

## Usage
```python
from src.svm import SVM
model = SVM(C=1.0, kernel='rbf')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```
