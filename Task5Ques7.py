print("Question 7")
str=input("Enetr a string: ")
length=len(str)
for i in range(0,length):
    print(i+1,"Character of string: ",str[i])

print("reverse String: ")
for i in range(len(str)-1, -1, -1):
    print(str[i], end="")

