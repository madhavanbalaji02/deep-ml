import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
	# Your code here
	m = len(y)

	for _ in range(n_iterations):
		if method == 'batch':

			predictions = X.dot(weights)
			errors = predictions - y
			gradient = 2 * X.T.dot(errors) / m
			weights = weights - learning_rate * gradient
		
		elif method == 'stochastic':
			for i in range(m):

				prediction = X[i].dot(weights)
				errors = prediction - y[i]
				gradient = 2 * X[i].T.dot(errors)
				weights = weights - learning_rate * gradient

		elif method == 'mini_batch':
			for i in range(0, m, batch_size):

				X_batch = X[i:i+batch_size]
				y_batch = y[i:i+batch_size]
				predictions = X_batch.dot(weights)
				errors = predictions - y_batch
				gradient = 2 * X_batch.T.dot(errors) / batch_size
				weights = weights - learning_rate * gradient
	return weights
