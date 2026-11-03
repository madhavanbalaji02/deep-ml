import numpy as np

def cross_product(a, b):
    a = np.array(a)
    b = np.array(b)

    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("Both input vectors must be of length 3.")

    cross = np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ])
    return cross