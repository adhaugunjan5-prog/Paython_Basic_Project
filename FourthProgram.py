print("Question No 7")
"""
This program calculates the area and perimeter of a rectangle.
It helps users quickly determine space coverage and boundary length.
"""

# User input makes the program flexible for different rectangle sizes.
length = float(input("Enter length: "))

# Width is required to perform rectangle calculations.
width = float(input("Enter width: "))

# Area indicates how much surface the rectangle covers.
area = length * width

# Perimeter represents the total distance around the rectangle.
perimeter = 2 * (length + width)

# Displaying results allows the user to verify the calculations.
print("Area =", area)

# Showing the perimeter separately improves output readability.
print("Perimeter =", perimeter)

print()
print("Question No 8")
fruits=["mango" , "apple" , "banana" , "cherry"]
score=50
score+=25
if "apple" in fruits :
    print("apple is present in list")          #print if apple is present in the list
else:
    print("apple is not present in list")      #print if apple is not present in the list

if "grape" not in fruits :
    print("grape is not present in list")      #print if grape is not  present in the list
else:
    print("grape is present in list")          #print if grape is present in the list