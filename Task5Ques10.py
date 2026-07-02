print("Question 10")
str=input("Enetr a string: ")
length=len(str)
print("The length of string: ",length)

mid = length // 2

print("First half:", str[:mid])
print("Second half:", str[mid:])

# Case-insensitive search
if "python" in str.lower():
    print("'python' is present.")
else:
    print("'python' is not present.")

print("\nCharacter Indices:")

for i in range(length):
    print("Positive Index:", i,
          "Negative Index:", i - length,
          "Character:", str[i])

print("\nReverse String:")
print(str[::-1])
