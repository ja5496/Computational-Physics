#Problem 7.2 From Newman

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#part A: make a graph of the sunspots vs. time:

df = pd.read_csv('sunspots.txt', delim_whitespace=True, names=['time', 'sunspots'])

plt.plot(df['time'], df['sunspots'])
plt.title('Sunspots', fontsize = 22, fontweight = 'bold')
plt.xlabel('Time [months]', fontsize = 20)
plt.ylabel('Sunspots', fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()

#Part B