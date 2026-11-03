import math

def normal_pdf(x, mean, std_dev):
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	"""
	# Your code here
	coeff = 1 / (math.sqrt(2 * math.pi) * std_dev)
    exponent = math.exp(-((x - mean) ** 2) / (2*std_dev** 2))
    return round(coeff * exponent, 5)