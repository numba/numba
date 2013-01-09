# Demo

## Linear Regression with Gradient Descent

# Pure Python + Numpy

```python
def gradient_descent(X, Y, theta, alpha, num_iters):
    m = Y.shape[0]
    theta_x = 0.0
    theta_y = 0.0
    for i in range(num_iters):
        predict = theta_x + theta_y * X
        err_x = (predict - Y)
        err_y = (predict - Y) * X
        theta_x = theta_x - alpha * (1.0 / m) * err_x.sum()
        theta_y = theta_y - alpha * (1.0 / m) * err_y.sum()
    theta[0] = theta_x
    theta[1] = theta_y
```

# Numba

```python
from numba import jit, f8, int32, void
@jit(void(f8[:], f8[:], f8[:], f8, int32))
def gradient_descent(X, Y, theta, alpha, num_iters):
    m = Y.shape[0]
    theta_x = 0.0
    theta_y = 0.0
    for i in range(num_iters):
        err_acc_x = 0.0
        err_acc_y = 0.0
        for j in range(X.shape[0]):
            predict = theta_x + theta_y * X[j]
            err_acc_x += predict - Y[j]
            err_acc_y += (predict - Y[j]) * X[j]
        theta_x = theta_x - alpha * (1.0 / m) * err_acc_x
        theta_y = theta_y - alpha * (1.0 / m) * err_acc_y
    theta[0] = theta_x
    theta[1] = theta_y
```

# NumbaPro

```python
import numbapro
from numba import jit, f8, int32, void
@jit(void(f8[:], f8[:], f8[:], f8, int32))
def gradient_descent(X, Y, theta, alpha, num_iters):
    m = Y.shape[0]
    theta_x = 0.0
    theta_y = 0.0
    for i in range(num_iters):
        predict = theta_x + theta_y * X
        err_x = (predict - Y)
        err_y = (predict - Y) * X
        theta_x = theta_x - alpha * (1.0 / m) * err_x.sum()
        theta_y = theta_y - alpha * (1.0 / m) * err_y.sum()
    theta[0] = theta_x
    theta[1] = theta_y
```
