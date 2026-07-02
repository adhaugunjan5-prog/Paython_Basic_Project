print("question 7")
# 1. Import the entire math module 
import math
# 2. Import only sqrt from math
from math import sqrt
# 3. Import math with an alias 
import math as m

print("2 power 4= ", math.pow(2,4))
print("square root of 25: ",math.sqrt(25)) 
print("factorialn of 5 : ", m.factorial(5))

def greet(name):
 print("Hello",name+", Welcome to python class! ")


def calculate_power(base,exp):
     return base**exp