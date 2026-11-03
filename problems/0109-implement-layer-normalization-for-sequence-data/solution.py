import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)
    X_mean = (X-mean) / np.sqrt(variance + epsilon)
    mean_X = gamma * X_mean + beta
    return mean_X
	