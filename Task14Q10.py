import numpy as np
print("Question 10: ")
# Generate marks of 10 students and 5 subjects
marks = np.random.randint(30,101,(10,5))

print("Student Marks:")
print(marks)

# Calculate total marks
total = marks.sum(axis=1)

# Calculate average marks
average = marks.mean(axis=1)

print("\nTotal Marks:")
print(total)

print("\nAverage Marks:")
print(average)

# Highest scorer
highest = total.argmax()

# Lowest scorer
lowest = total.argmin()

print("\nHighest Scoring Student Index:", highest)
print("Marks:", marks[highest])

print("\nLowest Scoring Student Index:", lowest)
print("Marks:", marks[lowest])

# Class statistics
print("\nOverall Class Mean:", marks.mean())
print("Overall Class Standard Deviation:", marks.std())

# Demonstrate reshape
reshaped = marks.reshape(5,10)

print("\nReshaped Array (5 x 10):")
print(reshaped)

# Top 3 students
top3 = total.argsort()[-3:][::-1]

print("\nTop 3 Students:")
print(marks[top3])

# Comments:
# sum(axis=1) -> Calculates total marks for each student.
# mean(axis=1) -> Calculates average marks.
# argmax() -> Finds highest scorer.
# argmin() -> Finds lowest scorer.
# argsort() -> Sorts totals and helps find top students.
# reshape() -> Changes array shape without changing data.

