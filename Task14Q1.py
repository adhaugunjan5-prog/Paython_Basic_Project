import numpy as np

# Create 2D array (5,6) with random integers between 10 and 100
arr = np.random.randint(10, 101, size=(5, 6))

# Print array
print("Array:")
print(arr)

# Array properties
print("\nShape of array:", arr.shape)
print("Total number of elements:", arr.size)
print("Data type:", arr.dtype)