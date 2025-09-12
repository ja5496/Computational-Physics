#Python file to solve the numerical integration problem from HW1 Q2
import numpy as np
import matplotlib.pyplot as plt

'''
A) First define integration functions. Use midpoint rule, trapezoidal rule, and Simpson's rule 
to compute the integral of each function from 0 to 1.

'''

def midpoint_rule(f, a, b, N):
    print('Running midpoint rule with N =', N)
    width = (b - a) / N
    integral = 0.0
    for i in range(N):
        x_mid = a + (i + 0.5) * width
        integral += f(x_mid).astype(np.float32)
    integral *= width
    return integral.astype(np.float32)

def trapezoidal_rule(f, a, b, N):
    print('Running trapezoidal rule with N =', N)
    width = (b - a) / N
    integral = 0.5 * (f(a) + f(b))
    for i in range(1, N):
        x_i = a + i * width
        integral += f(x_i).astype(np.float32)
    integral *= width
    return integral.astype(np.float32)

def simpsons_rule(f, a, b, N):
    print("Running Simpson's rule with N =", N)
    if N % 2 == 1:
        N += 1  # Simpson's rule requires an even number of intervals
    width = (b - a) / N
    integral = f(a) + f(b)
    for i in range(1, N, 2):
        x_i = a + i * width
        integral += 4 * f(x_i).astype(np.float32)
    for i in range(2, N-1, 2):
        x_i = a + i * width
        integral += 2 * f(x_i).astype(np.float32)
    integral *= width / 3
    return integral.astype(np.float32)

#Define parameters and functions to be used in the integration
x0, x1 = np.float32(0.0), np.float32(1.0)
N = np.logspace(0, 4, 100).astype(int) #Number of intervals to test
def func(x):
    return np.exp(-x).astype(np.float32)

vec_midpoint_rule = np.vectorize(midpoint_rule, excluded=['f', 'a', 'b'])
vec_trapezoidal_rule = np.vectorize(trapezoidal_rule, excluded=['f', 'a', 'b'])
vec_simpsons_rule = np.vectorize(simpsons_rule, excluded=['f', 'a', 'b'])
#Compute arrays of errors for each method:
real_val = (1-np.exp(-1)).astype(np.float32)
err_mid = (np.abs(vec_midpoint_rule(func, x0, x1, N) - real_val)/(real_val))
err_trap = (np.abs(vec_trapezoidal_rule(func, x0, x1, N) - real_val)/(real_val))
err_simp = (np.abs(vec_simpsons_rule(func, x0, x1, N) - real_val)/(real_val))

#Part B: Plot the errors for each method on a log-log scale:
plt.figure(figsize=(10, 8))
plt.loglog(N, err_mid, label='Midpoint Rule')
plt.loglog(N, err_trap, label='Trapezoidal Rule')
plt.loglog(N, err_simp, label="Simpson's Rule")
plt.title('Error in Numerical Integration of e^(-t)',fontsize=20, fontweight='bold')
plt.xlabel('Number of Bins (N)',fontsize=18)
plt.ylabel('Relative Error',fontsize=18)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()