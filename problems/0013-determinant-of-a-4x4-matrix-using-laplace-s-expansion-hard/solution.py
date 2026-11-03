def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here
	if len(matrix) == 1:
		return matrix[0][0]
	
	det = 0
	for c in range(len(matrix)):
		minor = [row[:c] + row[c+1:] for row in matrix[1:]]

		cofactor = ((-1)**c) * determinant_4x4(minor)
		det += matrix[0][c] * cofactor
	return det