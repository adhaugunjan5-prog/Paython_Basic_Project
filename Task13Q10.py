import numpy as np

try:
    # Taking input from user
    n = int(input("How many numbers do you want to generate? "))

    if n <= 0:
        print("Please enter a positive number.")
    else:
        # Generate random numbers between 10 and 100
        numbers = np.random.randint(10, 101, size=n)

        # Print generated array
        print("\nGenerated Array:")
        print(numbers)

        # Statistics
        print("\nMean:")
        print(np.mean(numbers))

        print("\nMedian:")
        print(np.median(numbers))

        print("\nStandard Deviation:")
        print(np.std(numbers))

        print("\nMinimum Value:")
        print(np.min(numbers))

        print("\nMaximum Value:")
        print(np.max(numbers))

        # Reshape into 2D array if possible
        rows = int(np.sqrt(n))

        if n % rows == 0:
            matrix = numbers.reshape(rows, n // rows)

            print("\nReshaped 2D Array:")
            print(matrix)

            # Row-wise sum
            print("\nRow-wise Sum:")
            print(np.sum(matrix, axis=1))

        else:
            print("\n2D reshape not possible for this number of elements.")

except ValueError:
    print("Invalid input! Please enter only numbers.")

except Exception as e:
    print("An error occurred:", e)