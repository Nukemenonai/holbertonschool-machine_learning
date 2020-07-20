#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

idx = ["Farrah", "Fred", "Felicia"]
fig, ax = plt.subplots()
ax.bar(idx, fruit[0],
       label="apples", color='r', width=0.5)

ax.bar(idx, fruit[1], bottom=fruit[0],
       label="bananas", color='yellow', width=0.5)

ax.bar(idx, fruit[2], bottom=fruit[0] + fruit[1],
       label="oranges", color='#ff8000', width=0.5)

ax.bar(idx, fruit[3], bottom=fruit[0] + fruit[1] + fruit[2],
       label="peaches", color='#ffe5b4', width=0.5)

ax.set_xticklabels(["Farrah", "Fred", "Felicia"])
ax.set_ylim(0, 80)
ax.set_yticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80])
ax.set_ylabel("Quantity of Fruit")
plt.legend()
plt.title("Number of Fruit per Person")
plt.show()
