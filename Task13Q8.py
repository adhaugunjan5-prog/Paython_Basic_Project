import numpy as np

# Create 4x4 matrix of random integers between 1 and 100
matrix = np.random.randint(1, 101, size=(4, 4))

# Printing matrix
print("Random 4x4 Matrix:")
print(matrix)

# Array properties
print("\nShape of array:")
print(matrix.shape)

print("\nDimension of array (ndim):")
print(matrix.ndim)

print("\nTotal number of elements (size):")
print(matrix.size)

print("\nData type:")
print(matrix.dtype)

print("\nMinimum value:")
print(matrix.min())

print("\nMaximum value:")
print(matrix.max())