import numpy as np

print("Question 9: ")
arr = np.random.randn(6,6)

print("Matrix:")
print(arr)

print("\nShape:", arr.shape)
print("Size:", arr.size)
print("Data Type:", arr.dtype)

print("\nIndex of Maximum Value:")
print(np.unravel_index(arr.argmax(), arr.shape))

print("\nIndex of Minimum Value:")
print(np.unravel_index(arr.argmin(), arr.shape))

print("\nTop Left 3x3 Matrix:")
print(arr[:3,:3])

# Replace negative numbers with absolute values
arr[arr < 0] = np.abs(arr[arr < 0])

print("\nModified Matrix:")
print(arr)

print("\nMean of Modified Matrix:")
print(arr.mean())





