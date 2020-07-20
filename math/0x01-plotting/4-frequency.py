#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, ax = plt.subplots()
ax.hist(student_grades, bins=[40, 50, 60, 70, 80, 90, 100], edgecolor ='k')
plt.xlim(0, 100)
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.yticks([0, 5, 10, 15, 20, 25, 30])
plt.title("Project A")
ax.set_xlabel("Grades")
ax.set_ylabel("Number of Students")
plt.show()
