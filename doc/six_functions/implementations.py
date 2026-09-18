"""CS-433 Project 1 — six required methods (NumPy only).

Every public function returns (w, loss) where w is 1D with shape (D,)
and loss is a scalar. Regularized methods return the unregularized loss.
"""

import numpy as np


def _mse(y, tx, w):
    """Mean squared error with the lecture factor 1 / (2N)."""
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def _mse_gradient(y, tx, w):
    """Gradient of MSE: (1/N) X^T (Xw - y)."""
    e = y - tx @ w
    return -(tx.T @ e) / y.shape[0]


def _sigmoid(z):
    """Numerically stable logistic sigmoid."""
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _logistic_loss(y, tx, w):
    """Mean negative log-likelihood for y in {0, 1}."""
    p = np.clip(_sigmoid(tx @ w), 1e-15, 1.0 - 1e-15)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _logistic_gradient(y, tx, w):
    """Gradient of mean NLL: (1/N) X^T (sigma(Xw) - y)."""
    return (tx.T @ (_sigmoid(tx @ w) - y)) / y.shape[0]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: shape (N,)
        tx: shape (N, D)
        initial_w: shape (D,)
        max_iters: number of GD steps
        gamma: step size

    Returns:
        (w, loss) last weights and MSE (factor 1/(2N))
    """
    y = np.asarray(y).reshape(-1)
    w = np.asarray(initial_w, dtype=float).reshape(-1)
    for _ in range(max_iters):
        w = w - gamma * _mse_gradient(y, tx, w)
    return w, _mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using SGD with mini-batch size 1.

    Args:
        y: shape (N,)
        tx: shape (N, D)
        initial_w: shape (D,)
        max_iters: number of SGD steps
        gamma: step size

    Returns:
        (w, loss) last weights and MSE on the full dataset
    """
    y = np.asarray(y).reshape(-1)
    w = np.asarray(initial_w, dtype=float).reshape(-1)
    n = y.shape[0]
    for _ in range(max_iters):
        i = np.random.randint(n)
        w = w - gamma * _mse_gradient(y[i : i + 1], tx[i : i + 1], w)
    return w, _mse(y, tx, w)


def least_squares(y, tx):
    """Least squares via the normal equations.

    Args:
        y: shape (N,)
        tx: shape (N, D)

    Returns:
        (w, loss) closed-form weights and MSE
    """
    y = np.asarray(y).reshape(-1)
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, _mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression via the normal equations.

    Penalty used in the solver: lambda_ * ||w||^2
    (together with MSE 1/(2N), this gives (X^T X + 2 N lambda_ I) w = X^T y).
    The returned loss does not include the penalty.

    Args:
        y: shape (N,)
        tx: shape (N, D)
        lambda_: regularization strength

    Returns:
        (w, loss)
    """
    y = np.asarray(y).reshape(-1)
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T @ y)
    return w, _mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Labels must be in {0, 1}.

    Args:
        y: shape (N,)
        tx: shape (N, D)
        initial_w: shape (D,)
        max_iters: number of GD steps
        gamma: step size

    Returns:
        (w, loss) last weights and mean NLL
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(initial_w, dtype=float).reshape(-1)
    for _ in range(max_iters):
        w = w - gamma * _logistic_gradient(y, tx, w)
    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression using gradient descent.

    Extra gradient term is 2 * lambda_ * w. Returned loss is unregularized NLL.

    Args:
        y: shape (N,), labels in {0, 1}
        tx: shape (N, D)
        lambda_: regularization strength
        initial_w: shape (D,)
        max_iters: number of GD steps
        gamma: step size

    Returns:
        (w, loss)
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(initial_w, dtype=float).reshape(-1)
    for _ in range(max_iters):
        grad = _logistic_gradient(y, tx, w) + 2.0 * lambda_ * w
        w = w - gamma * grad
    return w, _logistic_loss(y, tx, w)
