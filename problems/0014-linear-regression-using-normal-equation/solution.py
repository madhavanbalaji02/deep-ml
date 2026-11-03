import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float).reshape(-1, 1)

    theta = np.linalg.solve(X.T @ X, X.T @ y)
    theta = np.round(theta, 4).flatten().tolist()
	return theta