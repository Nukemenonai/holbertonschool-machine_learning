#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig = plt.figure
plt.suptitle(" All in One")

ax1 = plt.subplot2grid((3, 2), (0, 0))
ax2 = plt.subplot2grid((3, 2), (0, 1))
ax3 = plt.subplot2grid((3, 2), (1, 0))
ax4 = plt.subplot2grid((3, 2), (1, 1))
ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

ax1.plot(y0, color='r')
ax1.set_xticks([0, 2, 4, 6, 8, 10])
ax1.set_yticks([0, 500, 1000])

ax2.scatter(x1, y1, c="m")
ax2.set_xlabel("Height (in)")
ax2.set_ylabel("Weight (lbs)")
ax2.set_title("Men's Height vs Weight")

ax3.plot(x2, y2)
ax3.set_yscale('log')
ax3.set_xlim(0, 28650)
ax3.set_xlabel("Time (years)")
ax3.set_ylabel("Fraction remaining")
ax3.set_title("Exponential Decay of C-14")

ax4.plot(x3, y31, color='r', linestyle='--')
ax4.plot(x3, y32, color='g')
ax4.set_xlim(0, 20000)
ax4.set_ylim(0, 1)
ax4.set_xlabel("Time (years)")
ax4.set_ylabel("Fraction Remaining")
ax4.legend(('C-14', 'Ra-226'), loc='upper right')
ax4.set_title("Exponential Decay of Radioactive Elements")

ax5.hist(student_grades, bins=[40,50,60,70,80,90,100], edgecolor ='k')
#ax5.xlim(0,100)
ax5.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax5.set_yticks([0, 10, 20, 30])
ax5.set_title("Project A")
ax5.set_xlabel("Grades")
ax5.set_ylabel("Number of Students")

plt.tight_layout()
plt.show()
