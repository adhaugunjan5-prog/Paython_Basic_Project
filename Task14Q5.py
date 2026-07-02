import numpy as np
print("Question 5: ")
arr = np.array([[10,20,30,40],
                [50,60,70,80],
                [90,100,110,120]])

print("Original Array:")
print(arr)

# First row
print("\nFirst Row:")
print(arr[0])

# Last column
print("\nLast Column:")
print(arr[:,3])

# Center 2x2 submatrix
print("\nCenter 2x2 Submatrix:")
print(arr[1:3,1:3])

# Even numbers
print("\nEven Numbers:")
print(arr[arr % 2 == 0])

import numpy as np
