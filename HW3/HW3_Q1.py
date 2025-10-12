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

#Part B: Calculate Fourier Transform and plot the power spectrum

time = df['time'].to_numpy()
sunspots = df['sunspots'].to_numpy()

fft_vals = np.fft.rfft(sunspots)
f = np.fft.rfftfreq(len(sunspots), d=1)       # cycles per unit time
k = 2*np.pi*f               

power = np.abs(fft_vals)**2

plt.plot(f, power, linewidth=2.0)
plt.title('Power Spectrum', fontsize=20)
plt.xlabel('k [rad / time]', fontsize=18)
plt.ylabel('Power |C_k|^2', fontsize=18)
plt.ylim(-10,3e9)
plt.xscale('log')
plt.show()
