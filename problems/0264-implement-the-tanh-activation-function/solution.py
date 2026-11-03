import math

def tanh(x: float) -> float:
	"""
	Implements the Tanh (hyperbolic tangent) activation function.

	Args:
		x (float): Input value

	Returns:
		float: The tanh of the input, rounded to 4 decimal places
	"""
	# Your code here
	e_x = math.exp(x)
    e_neg_x = math.exp(-x)
    res = (e_x - e_neg_x) / (e_x + e_neg_x)
    return round(res, 4)