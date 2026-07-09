print("Question No 9")
name=input("Enter name : ")                         #take name input from user 
age = int(input("Enter age : "))                    #take age input from user 
city=input("Enter city : ")                         #take city input from user 
sub1 = float(input("Enter marks of subject 1 : "))  #take marks of subject 1  input from user 
sub2 = float(input("Enter marks of subject 2 : "))  #take marks of subject 2  input from user 
sub3 = float(input("Enter marks of subject 3 : "))  #take marks of subject 3  input from user 
total=sub1+sub2+sub3                                #calculate the total of all subjects
percentage=total/3                                  #calculate percentage

#print student profile with all information
print("......Student Profile.....")
print("name: ",name)
print("age: ",age)
print("city: ",city)
print("total: ",total)
print("percentage: ",percentage)

print()
print("Question No 10")
name ="Alice"                     #Name → changed to name (variable names are case-sensitive).
age=20                            #age 20 → changed to age = 20 (missing assignment operator).
city ="Amsterdam"                 #city = Amsterdam → changed to city = "Amsterdam" (strings need quotes).
print("My name is" + name)  
print("I am" ,age ,"years old")   #print("I am" age "years old") → commas added between values.
print("I live in" ,city)          # print("I live in " city) → comma added.
#Check if age is greater than 18 and print message
print("Adult:" ,age > 18)         #print("Adult:" age > 18) → comma added before the condition.