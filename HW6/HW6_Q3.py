import numpy as np
import matplotlib.pyplot as plt

# Global Constants for LCG 
A = 1664525
C = 1013904223
M = 2**32 
N_POINTS = 10000 # Number of data points to generate

# Part A: Linear Congruential Generator (LCG)
def lcg_generate(seed, n_samples):
    """
    Generates n_samples of uniform random numbers in [0, 1)
    using the Linear Congruential method: X_{n+1} = (aX_n + c) mod m
    """
    numbers = []
    current = seed
    for _ in range(n_samples):
        current = (A * current + C) % M
        numbers.append(current / M)
    return np.array(numbers)

# Part B: Gaussian Generator (Box-Muller Transform)
def gaussian_from_uniform(u):
    """
    Transforms uniform random variables into Gaussian distributed variables
    using the Box-Muller transform.
    """
    half_n = len(u) // 2
    u1 = u[:half_n]
    u2 = u[half_n:]
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return z

# Part C & E: Power Spectrum Calculation
def calculate_power_spectrum(data):
    """
    Calculates the power spectrum |c_k|^2 using FFT.
    """
    c_k = np.fft.rfft(data)
    power = np.abs(c_k)**2
    k = np.fft.rfftfreq(len(data)) * len(data) 
    return k, power

print("Running simulation...")

# 1. Generate Uniform Random Numbers (LCG)
seed = 42
uniforms = lcg_generate(seed, N_POINTS * 2)

# 2. Generate Gaussian Random Numbers
gaussians = gaussian_from_uniform(uniforms)
gaussians = gaussians[:N_POINTS] 

# 3. Construct Random Walk (Cumulative Sum)
random_walk = np.cumsum(gaussians)

# 4. Calculate Power Spectra
k_gauss, P_gauss = calculate_power_spectrum(gaussians)
k_walk, P_walk = calculate_power_spectrum(random_walk)

print("Calculations complete. Preparing plots...")

# Plots:

# Create a layout: 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Common font sizes
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12

# Plot A: Histogram of Gaussian vs Unit Gaussian
ax1 = axes[0, 0]
counts, bins, _ = ax1.hist(gaussians, bins=50, density=True, alpha=0.6, label='LCG Data', color='blue')
x_th = np.linspace(-4, 4, 100)
y_th = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_th**2)
ax1.plot(x_th, y_th, 'r-', lw=2.5, label='Unit Gaussian')

ax1.set_yscale('log')
ax1.set_title("Gaussian Distribution (Log Scale)", fontsize=TITLE_SIZE, fontweight='bold')
ax1.set_xlabel("Value", fontsize=LABEL_SIZE)
ax1.set_ylabel("Probability Density (Log)", fontsize=LABEL_SIZE)
ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)

# Plot B: Random Walk Path
ax2 = axes[0, 1]
ax2.plot(random_walk, lw=1.5, color='purple')
ax2.set_title("Random Walk Trajectory", fontsize=TITLE_SIZE, fontweight='bold')
ax2.set_xlabel("Iteration Step $i$", fontsize=LABEL_SIZE)
ax2.set_ylabel("Position $x_i$", fontsize=LABEL_SIZE)
ax2.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax2.grid(True, linestyle='--', alpha=0.5)

# Plot C: Power Spectrum of Gaussian Noise
ax3 = axes[1, 0]
ax3.loglog(k_gauss[1:], P_gauss[1:], lw=1, color='green')
ax3.set_title("Power Spectrum: Gaussian Noise", fontsize=TITLE_SIZE, fontweight='bold')
ax3.set_xlabel("Wavenumber $k$", fontsize=LABEL_SIZE)
ax3.set_ylabel("Power $P(k)$", fontsize=LABEL_SIZE)
ax3.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax3.grid(True, which="both", linestyle='--', alpha=0.5)

# Plot D: Power Spectrum of Random Walk
ax4 = axes[1, 1]
ax4.loglog(k_walk[1:], P_walk[1:], lw=1.5, color='orange', label='Random Walk')

# Add 1/k^2 reference line
ref_k = k_walk[1:]
ref_P = 1e8 * (ref_k**-2) 
ax4.loglog(ref_k, ref_P, 'k--', lw=2, label=r'Scaling $k^{-2}$')

ax4.set_title("Power Spectrum: Random Walk", fontsize=TITLE_SIZE, fontweight='bold')
ax4.set_xlabel("Wavenumber $k$", fontsize=LABEL_SIZE)
ax4.set_ylabel("Power $P(k)$", fontsize=LABEL_SIZE)
ax4.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax4.legend(fontsize=12)
ax4.grid(True, which="both", linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
