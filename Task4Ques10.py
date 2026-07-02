print("question 10")

import math
import random

# Lambda function to calculate square
square = lambda x: x * x

# Function to calculate power
def calculate_power(base, exponent):
    return math.pow(base, exponent)

# Function to generate random number
def generate_random():
    return random.randint(1, 100)

# Main program
while True:
    print("\n===== Simple Math Utility =====")
    print("1. Square")
    print("2. Power")
    print("3. Random Number")
    print("4. Exit")

    choice = input("Enter your choice (1-4): ")

    # Square
    if choice == "1":
        num = float(input("Enter a number: "))
        print("Square =", square(num))

    # Power
    elif choice == "2":
        base = float(input("Enter base: "))
        exponent = float(input("Enter exponent: "))
        print("Result =", calculate_power(base, exponent))

    # Random Number
    elif choice == "3":
        print("Random Number =", generate_random())

    # Exit
    elif choice == "4":
        break

    # Invalid choice
    else:
        print("Invalid choice! Please try again.")