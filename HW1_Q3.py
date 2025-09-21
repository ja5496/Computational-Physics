import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# --- 1) Load & sanity-filter (must be > 0 before logs) ---
df = pd.read_csv("lcdm_z0.matter_pk", sep=r"\s+", usecols=[0, 1], header=None)
df.columns = ["k", "P(k)"]
df = df[(df["k"] > 0) & (df["P(k)"] > 0)].copy()

k_data = df["k"].values
P_data = df["P(k)"].values

# Log–log clamped spline returning a callable P(k) ---
def cspline_interpolator(k, P, n_lo=0.4, n_hi=-1.3):
    # Build spline in log–log space so end slopes are d ln P / d ln k
    x = np.log(k)
    y = np.log(P)
    cs = CubicSpline(x, y, bc_type=((1, n_lo), (1, n_hi)), extrapolate=True)
    def P_of_k(k_eval):
        ke = np.asarray(k_eval)
        return np.exp(cs(np.log(ke)))
    return P_of_k

P_k = cspline_interpolator(k_data, P_data)

# Integrand for E(r) 
def integrand(k, r_i):
    # E(r) integrand: (1/2π^2) k^2 P(k) sin(kr)/(kr)
    return (1.0/(2.0*np.pi**2)) * k**2 * P_k(k) * (np.sin(k*r_i)/(k*r_i))

#Simpson's rule 
def simpsons(f, a, b, N, r):
    if N % 2 == 1:
        N += 1
    h = (b - a) / N
    s = f(a, r) + f(b, r)
    for i in range(1, N, 2):
        s += 4.0 * f(a + i*h, r)
    for i in range(2, N, 2):
        s += 2.0 * f(a + i*h, r)
    return s * h / 3.0

# Integration limits
a = k_data.min()
b = 50

# Compute E(r) 
r = np.linspace(0.0, 120.0, 1000)
E_r = np.array([simpsons(integrand, a, b, 3500, ri) for ri in r])
E_r_residual = np.array([simpsons(integrand, 50, 100, 3000, ri) for ri in r])
E_err = E_r_residual/E_r #Relative error

#Plot for Interpolated Power function
plt.figure(figsize=(8,6))
plt.loglog(k_data, P_k(k_data), linestyle="-")
plt.xlabel("Wavenumber (k)", fontsize=20)
plt.ylabel("P(k)", fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Interpolated P(k)", fontsize=24)
plt.legend()
plt.grid(True)
plt.show()

# Plot for Correlation Function:
plt.figure(figsize=(8,6))
plt.plot(r, r**2 * E_r, linestyle="-")
plt.xlabel("r [Mpc/h]", fontsize=20)
plt.ylabel(r"$r^2 E(r)$", fontsize=20)
plt.axvspan(95, 120, color="red", alpha=0.3, label="BAO Length Scale")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Correlation Function from Interpolated P(k)", fontsize=24)
plt.legend()
plt.grid(True)
plt.show()

#Plot For Relative Error Convergence Test:
plt.figure(figsize=(8,6))
plt.plot(k_data, E_err, linestyle="-")
plt.xlabel("Distance r", fontsize=20)
plt.ylabel("Relative Error in E(r)", fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Relative Error From Doubling Upper Bound", fontsize=24)
plt.legend()
plt.grid(True)
plt.show()