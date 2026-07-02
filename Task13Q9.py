import numpy as np

# Generate 1D array of 20 random integers between 1 and 50
arr = np.random.randint(1, 51, size=20)

# Reshape into 4x5 matrix
matrix = arr.reshape(4, 5)

# Printing matrix
print("4x5 Matrix:")
print(matrix)

# Sum, mean and standard deviation
print("\nSum of matrix:")
print(np.sum(matrix))

print("\nMean of matrix:")
print(np.mean(matrix))

print("\nStandard Deviation of matrix:")
print(np.std(matrix))

# Maximum value in each row
print("\nMaximum value in each row:")
print(np.max(matrix, axis=1))