"""
six functions
"""

import numpy as np

#  Mean squared error with the factor 1 / (2N)
def mse(y,tx,w):
    """MSE loss with factor 1/(2N)."""
    e = y - tx @ w
    out = 0.5 * np.mean(e**2)
    return out

#  Mean squared error gradient
def mse_gradient(y,tx,w):
    """Gradient of the MSE loss."""
    e = y - tx @ w
    out = -(tx.T @ e) / y.shape[0]
    return out

# mse gradient descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent. Returns (w, loss)."""
    w = np.asarray(initial_w, dtype=float).reshape(-1)
    y = np.asarray(y).reshape(-1)
    for _ in range(max_iters):
        w = w - gamma * mse_gradient(y, tx, w)
    return w, mse(y, tx, w)

# mse sgd
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using SGD with batch size 1. Returns (w, loss)."""
    w = np.asarray(initial_w, dtype=float).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = y.shape[0]
    for _ in range(max_iters):
        i = np.random.randint(n)
        w = w - gamma * mse_gradient(y[i:i+1], tx[i:i+1],w)
    return w, mse(y, tx, w)

# least squares
def least_squares(y, tx):
    """Least squares via the normal equations. Returns (w, loss)."""
    y = np.asarray(y).reshape(-1)
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, mse(y, tx, w)

# ridge regression
def ridge_regression(y, tx, lambda_):
    """Ridge regression via the normal equations. Returns (w, unregularized MSE)."""
    y = np.asarray(y).reshape(-1)
    n,d = tx.shape
    a = tx.T @ tx + 2*n*lambda_*np.eye(d)
    w = np.linalg.solve(a, tx.T@y)
    return w, mse(y, tx, w)

# logistic regression - binary classifacation
def sigmoid(z):
    """Logistic sigmoid, clipped for numerical stability."""
    z = np.clip(z,-30,30)
    return 1.0 / (1.0 + np.exp(-z))

def logistic_loss(y,tx,w):
    """Mean negative log-likelihood for labels in {0, 1}."""
    p = np.clip(sigmoid(tx @ w), 1e-15, 1 - 1e-15)
    return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))

def logistic_gradient(y,tx,w):
    """Gradient of the mean negative log-likelihood."""
    return (tx.T @ (sigmoid(tx@w)-y))/y.shape[0]

def logistic_regression(y,tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent. y in {0, 1}. Returns (w, loss)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(initial_w, dtype=float).reshape(-1)
    for _ in range(max_iters):
        w = w - gamma * logistic_gradient(y, tx, w)
    return w, logistic_loss(y, tx, w)

# logistic regression
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression using GD. Returns (w, unregularized loss)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(initial_w, dtype=float).reshape(-1)
    for _ in range(max_iters):
        grad = logistic_gradient(y, tx, w) + 2.0 * lambda_ * w
        w = w - gamma * grad
    return w, logistic_loss(y, tx, w)