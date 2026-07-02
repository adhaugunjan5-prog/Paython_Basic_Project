import numpy as np

# Creating two matrices
A = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

B = np.array([[9,8,7],
              [6,5,4],
              [3,2,1]])

# Matrix addition
addition = A + B

# Matrix multiplication
matrix_multiplication = A @ B

# Element-wise multiplication
element_multiplication = A * B

# Printing results
print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

print("\nMatrix Addition:")
print(addition)

print("\nMatrix Multiplication:")
print(matrix_multiplication)

print("\nElement-wise Multiplication:")
print(element_multiplication)