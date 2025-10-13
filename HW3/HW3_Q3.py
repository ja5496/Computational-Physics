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

def deconvolve(blurry_array, gauss_periodic=gauss_periodic):
    #Calculate the point spread function: 
    size = len(blurry_array[0])
    point_spread = np.fromfunction(gauss_periodic, (size, size), dtype=float) 

    s = point_spread.sum()
    if s != 0:
        point_spread = point_spread / s

    #Take the fourier transform of both the blurry array and the point spread func
    blur_fft = np.fft.rfft2(blurry_array)
    ps_ffts = np.fft.rfft2(point_spread)

    #Divide the blurry fft coeffs by the point spead function fft times (size)^2 to get the unblurred coeffs
    unblurred_fft = np.empty_like(blur_fft, dtype=blur_fft.dtype)
    for i, j in np.ndindex(blur_fft.shape):
        if np.abs(ps_ffts[i, j]) > 1e-6:
            unblurred_fft[i,j] = blur_fft[i,j]/(ps_ffts[i,j])
        else: 
            unblurred_fft[i,j] = blur_fft[i,j]

    #Perform inverse transform to get unblurred array:
    unblurred_arr = np.fft.irfft2(unblurred_fft)

    #Plot unblurred photo:
    plt.figure(figsize=(5,5))
    plt.imshow(unblurred_arr, cmap='gray', aspect='auto')
    plt.title('Deconvolved Density Plot', fontsize = 20, fontweight='bold')
    plt.axis('off')
    plt.show()


deconvolve(blurry_array)





