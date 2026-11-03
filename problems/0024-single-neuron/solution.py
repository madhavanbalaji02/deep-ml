import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	probabilities = []
	for feature_vector in features:
		z = sum(weight * feature for weight, feature in zip(weights, feature_vector)) + bias
		prob = 1 / (1 + math.exp(-z))
		probabilities.append(round(prob, 4))

	mse = sum((prob - label) ** 2 for prob, label in zip(probabilities, labels)) / len(labels)
	mse = round(mse, 4)

	return probabilities, mse
