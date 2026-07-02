import numpy as np

# a) 1D array of 10 random numbers between 0 and 1
arr1 = np.random.rand(10)

# b) 3x3 matrix of random numbers from standard normal distribution
arr2 = np.random.randn(3, 3)

# c) 2D array (4x5) of random integers between 10 and 50
arr3 = np.random.randint(10, 51, size=(4, 5))

# Printing arrays with headings
print("1D Array of 10 random numbers (0 to 1):")
print(arr1)

print("\n3x3 Matrix of random numbers (standard normal distribution):")
print(arr2)

print("\n4x5 Array of random integers (10 to 50):")
print(arr3)