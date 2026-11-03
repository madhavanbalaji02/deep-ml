import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)      # (n_samples,)
    n, d = X.shape                     # here: (3000, 8)

    # Closed-form least squares: W* = (X^T X + λI)^(-1) X^T y
    # small λ for numerical stability (ridge-like, but almost OLS) [web:35][web:38]
    lam = 1e-6
    XtX = X.T @ X                      # (8, 8)
    A = XtX + lam * np.eye(d)          # (8, 8)
    Xty = X.T @ y                      # (8,)

    W_trained = np.linalg.solve(A, Xty)  # (8,)

    # For standardized X and y, the optimal intercept is 0. [web:37][web:40]
    b_trained = 0.0

    # Ensure exact required types/shapes
    W_trained = np.asarray(W_trained, dtype=X.dtype).reshape(d,)
    b_trained = float(b_trained)

    return W_trained, b_trained

