import numpy as np

# a) Create 1D array
arr1 = np.array([10, 20, 30, 40, 50])

# b) Create 2D array (3x3)
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Print 1D array details
print("1D Array:")
print(arr1)
print("Shape:", arr1.shape)
print("Data Type:", arr1.dtype)

print()

# Print 2D array details
print("2D Array:")
print(arr2)
print("Shape:", arr2.shape)
print("Data Type:", arr2.dtype)