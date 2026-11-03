def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    trace = a + d
    det = a*d - b*c
    disc = trace**2 - 4*det

    lambda_1 = (trace + disc**0.5) / 2
    lambda_2 = (trace - disc**0.5) / 2

    return [lambda_1, lambda_2]

