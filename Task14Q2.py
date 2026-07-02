import numpy as np

# Generate 1D array of 20 random numbers between 1 and 50
arr = np.random.randint(1, 51, size=20)

# Print array
print("Array:")
print(arr)

# Statistical methods
print("\nMinimum value:", arr.min())
print("Minimum value index (argmin):", arr.argmin())

print("\nMaximum value:", arr.max())
print("Maximum value index (argmax):", arr.argmax())

print("\nSum of all elements:", arr.sum())
print("Mean:", arr.mean())
print("Standard Deviation:", arr.std())