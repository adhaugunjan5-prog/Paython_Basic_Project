print("Question No 5")
name=input("enter your name: ")                     # take name input from user 
birth_year = int(input("Enter your Birth Year: "))  # because input function take input string by default(need to convert string to int)
cur_year=2026                                       # initilization of cur_year variable(cur_year is store the value of current year(that is 2026))
cur_age=cur_year-birth_year                         # initilization of cur_age variable(cur_age(that is cur_year-birth_year))
print("current age of ", name , "is" , cur_age)     # print the current age

print()
print("Question No 6")
height = float(input("Enter height in meter: "))     # take height input from user 
weight = float(input("Enter weight in kg: "))        # take weight input from user 

bmi=(weight/(height**2))                             # calculate the BMI using  bmi=(weight/(height**2))  formula
print("BMI: ",round(bmi,2))                          # print the BMI value with round 2 decimal