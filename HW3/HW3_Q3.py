#Newman Excercise 7.9

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Part A: Read in the grid and 

df = pd.read_csv('blur.txt', sep='\s+', header=None)
blurry_array = df.to_numpy()

plt.figure(figsize=(5,5))
plt.imshow(blurry_array, cmap='gray', aspect='auto')
plt.title('Blurry Density Plot', fontsize = 20, fontweight='bold')
plt.axis('off')
plt.show()

#Part B: Create a grid of samples drawn from Gaussian function 

size = len(blurry_array[0]) #the array is square 
sigma = 25.0
ic, jc = 0, 0    # center at (0,0) 

def gauss_periodic(i, j, N=size, ic=ic, jc=jc, sigma=sigma):
    di = np.minimum(np.abs(i - ic), N - np.abs(i - ic))
    dj = np.minimum(np.abs(j - jc), N - np.abs(j - jc))
    return np.exp(-(di**2 + dj**2) / (2*sigma**2))

gaussian_density = np.fromfunction(gauss_periodic, (size, size), dtype=float) #initialize the size of the array.

plt.figure(figsize=(5,5))
plt.imshow(gaussian_density, cmap='gray', aspect='auto')
plt.title('Gaussian Density Plot', fontsize = 20, fontweight='bold')
plt.axis('off')
plt.show()

#Part C: Deconvolution function



