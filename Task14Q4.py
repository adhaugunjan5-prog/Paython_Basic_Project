import numpy as np

# Create a 1D array from 1 to 24
arr = np.arange(1, 25)

print("Original 1D Array:")
print(arr)
print("Shape:", arr.shape)

# Reshape into (4, 6)
arr1 = arr.reshape(4, 6)
print("\nReshaped Array (4, 6):")
print(arr1)
print("Shape:", arr1.shape)

# Reshape into (3, 8)
arr2 = arr.reshape(3, 8)
print("\nReshaped Array (3, 8):")
print(arr2)
print("Shape:", arr2.shape)

# Reshape into (2, 3, 4)
arr3 = arr.reshape(2, 3, 4)
print("\nReshaped Array (2, 3, 4):")
print(arr3)
print("Shape:", arr3.shape)
