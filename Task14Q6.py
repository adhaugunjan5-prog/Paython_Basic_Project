import numpy as np
arr = np.random.randint(1,101,(5,5))

print("Original Matrix:")
print(arr)

# Diagonal elements
print("\nDiagonal Elements:")
print(np.diag(arr))

# Elements greater than 50
print("\nElements Greater Than 50:")
print(arr[arr > 50])

# Replace values less than 30 with 0
arr[arr < 30] = 0

print("\nModified Matrix:")
print(arr)



import numpy as np
