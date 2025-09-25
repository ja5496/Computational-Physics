# Python file to solve the numerical integration problem from HW1 Q2
import numpy as np
import matplotlib.pyplot as plt

# Shorthand for making things into single precision because I will have to do it a lot
f32 = np.float32

# Define my integration rules
def midpoint_rule(f, a, b, N):
    a = f32(a); b = f32(b)
    width = (b - a) / f32(N)
    integral = f32(0.0)
    for i in range(N):
        x_mid = a + (f32(i) + f32(0.5)) * width
        integral = f32(integral + f(x_mid))
    return f32(integral * width)

def trapezoidal_rule(f, a, b, N):
    a = f32(a); b = f32(b)
    width = (b - a) / f32(N)
    integral = f32(0.5) * f32(f(a) + f(b))
    for i in range(1, N):
        x_i = a + f32(i) * width
        integral = f32(integral + f(x_i))
    return f32(integral * width)

def simpsons_rule(f, a, b, N):
    a = f32(a); b = f32(b)
    if N % 2 == 1:
        N += 1  # Simpson's rule requires even N
    width = (b - a) / f32(N)
    integral = f32(f(a) + f(b))
    four = f32(4.0); two = f32(2.0)
    for i in range(1, N, 2):
        x_i = a + f32(i) * width
        integral = f32(integral + four * f(x_i))
    for i in range(2, N - 1, 2):
        x_i = a + f32(i) * width
        integral = f32(integral + two * f(x_i))
    return f32(integral * (width * f32(1.0/3.0)))

# Bounds of Integration
x0, x1 = f32(0.0), f32(1.0)

# N values from 0 to 10^7 to reach machine precision for bin sizes to see both truncation and roundoff regimes
N_values = np.unique(np.logspace(0, 7, 100).astype(int))  

# define the integrand e^{-t} in float32
def integrand(x_f32):
    # we pass x already as float32; compute exp(-x) in float32
    return np.exp(-f32(x_f32)).astype(np.float32)

# exact analytic solution to the integral fro calculating the error
real_val = f32(1.0) - np.exp(f32(-1.0)).astype(np.float32)

# Compute errors by looping over N 
err_mid, err_trap, err_simp = [], [], []
for N in N_values:
    print("Evaluating at N = ", N)
    I_mid  = midpoint_rule(integrand, x0, x1, int(N))
    I_trap = trapezoidal_rule(integrand, x0, x1, int(N))
    I_simp = simpsons_rule(integrand, x0, x1, int(N))

    e_mid  = f32(np.abs(f32(I_mid  - real_val)) / real_val)
    e_trap = f32(np.abs(f32(I_trap - real_val)) / real_val)
    e_simp = f32(np.abs(f32(I_simp - real_val)) / real_val)

    err_mid.append(e_mid)
    err_trap.append(e_trap)
    err_simp.append(e_simp)

err_mid  = np.array(err_mid,  dtype=np.float32)
err_trap = np.array(err_trap, dtype=np.float32)
err_simp = np.array(err_simp, dtype=np.float32)

# Now plot finally
plt.figure(figsize=(10, 8))
plt.loglog(N_values, err_mid,  label='Midpoint')
plt.loglog(N_values, err_trap, label='Trapezoidal')
plt.loglog(N_values, err_simp, label="Simpson's")
plt.loglog(N_values, 1/N_values, linestyle=":", color = 'black')
plt.loglog(N_values, 1/(N_values**(2)), linestyle=":", color = 'black')
plt.loglog(N_values, 1/(N_values**(4)), linestyle=":", color = 'black')
plt.title('Numerical Integration of e^{-t}', fontsize=24, fontweight='bold')
plt.xlabel('Number of Bins (N)', fontsize=22)
plt.ylabel('Relative Error', fontsize=22)
plt.ylim(bottom = 0.5e-7)
plt.ylim(top = 5e-1)
plt.legend(fontsize=16)
plt.grid(True)
plt.show()