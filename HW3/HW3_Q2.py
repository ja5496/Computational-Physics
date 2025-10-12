#Newman 7.4 on Fourier Transforms

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Part A: Read in values and plot dow data.

df = pd.read_csv('dow.txt', delim_whitespace=True, names=['dow'])
dow_data = df['dow'].to_numpy()
days = [i for i in range(len(df['dow'].to_numpy()))]

plt.plot(days, dow_data)
plt.title('Stock Market Data', fontsize = 22, fontweight = 'bold')
plt.xlabel('Time [days]', fontsize = 20)
plt.ylabel('Dow Jones Close [$]', fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid(True)
plt.show()

#Part B: Calculate Fourier Coefficients.

coeffs = np.fft.rfft(dow_data)

#Part C: Set a percentage of the coefficients to zero.

def set_to_zero(arr, fraction_keep): #inputs are the array, and the fraction of it to not set to 0
    keep_index = int(fraction_keep*len(arr))
    new_arr = np.append(arr[:keep_index], [0 for i in arr[keep_index:]])
    return new_arr

ten_percent_coeffs = set_to_zero(coeffs, 0.10)
two_percent_coeffs = set_to_zero(coeffs, 0.02)

#Part D: Inverse Fourier Transform.

dow_ten = np.fft.irfft(ten_percent_coeffs, n=len(dow_data))
dow_two = np.fft.irfft(two_percent_coeffs, n=len(dow_data))

plt.plot(days, dow_data, label='Original', linewidth=2)
plt.plot(days, dow_ten, label='First 10% Coeffs Only', linewidth=2)
plt.plot(days, dow_two, label='First 2% Coeffs Only', linewidth=2)
plt.title('Stock Market Data', fontsize = 22, fontweight = 'bold')
plt.xlabel('Time [days]', fontsize = 20)
plt.ylabel('Dow Jones Close [$]', fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid(True)
plt.legend()
plt.show()