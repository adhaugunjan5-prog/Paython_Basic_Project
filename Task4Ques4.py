print("question 4")
total = 0

while True:
    num = int(input("Enter a positive number: "))

    if num <= 0:
        break

    total += num

print("Total Sum =", total)
