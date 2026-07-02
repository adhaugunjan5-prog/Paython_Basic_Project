print("Question 8")
num = int(input("Enter a number: "))

if num in range(1, 51):
    print(num, "is present in range(1,51)")
else:
    print(num, "is not present in range(1,51)")

if num in range(10, 100, 5):
    print(num, "is present in range(10,100,5)")
else:
    print(num, "is not present in range(10,100,5)")
