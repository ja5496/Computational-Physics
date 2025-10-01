#Code for Question 3 on HW2: Implementation of gradient descent to fit a Schecter function to data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

def f(x,y):
    return (x-2)**2+(y-2)**2

def grad_descent_2D(func, gamma, h, x_0, y_0, tolerance):
    #need an initial h to help the process get started since we need two points for future numerical derivs:
    partialx_0 = (func(x_0,y_0-h)-func(x_0-h,y_0-h))/h
    partialy_0 = (func(x_0-h,y_0)-func(x_0-h,y_0-h))/h
    
    x_i = x_0 - gamma*partialx_0
    y_i = y_0 - gamma*partialy_0

    step_size = np.sqrt((gamma*partialx_0)**2+(gamma*partialy_0)**2)
    i = 1
    # keep track of past steps for plotting 
    x_step_memory = [dc(x_0),dc(x_i)]
    y_step_memory = [dc(y_0),dc(y_i)]

    while step_size > tolerance: 
        i+=1
        print(i)
        partialx = (func(x_i,y_0)-func(x_0,y_0))/(x_i-x_0)
        partialy = (func(x_0,y_i)-func(x_0,y_0))/(y_i-y_0)

        #Take step and get coordinates ready for next numerical derivative.
        x_0 = dc(x_i)
        y_0 = dc(y_i)
        x_i += - gamma * partialx
        y_i += - gamma * partialy

        x_step_memory.append(dc(x_i))
        y_step_memory.append(dc(y_i))

        step_size = np.sqrt((gamma*partialx)**2+(gamma*partialy)**2)

    x_final = x_i
    y_final = y_i

    return x_final, y_final, x_step_memory, y_step_memory

x_final, y_final, x_arr, y_arr = grad_descent_2D(f,0.1,0.1,1,1,1e-5)

x = np.linspace(0.5, 3, 400)
y = np.linspace(0.5, 3, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, levels=15, cmap="viridis")  # 10 contour levels
plt.plot(x_arr,y_arr, marker = 'o')
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.title('Gradient Descent Test', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.show()

#Now we need to use gradient descent on the chi^2 value of a fit function 

df = pd.read_csv("smf_cosmos.dat", delim_whitespace=True, header=None, names=["log_M_gal", "n", "err"])
masses = df["log_M_gal"]
n_data = df["n"]
sigma = df["err"]

#Now define Schecter Function:
def n(M_gal, phi_star, M_star , a):
    #convert to exp scale. An input value for M_gal = 3 will be 10^3 for example
    M_star_ = 10**M_star
    phi_star_ = 10**phi_star
    return phi_star_*((10**M_gal)/M_star_)**(a+1)*np.exp(- (10**M_gal)/M_star_)*np.log(10)

def chi_squared(phi_star, M_star , a):
    chi_sq = 0
    for i in range(len(df["log_M_gal"])):
        chi_sq+=(n(masses[i], phi_star, M_star, a)-n_data[i])**2/(sigma[i]**2)
    return chi_sq

def grad_descent_3D(func, gamma, h, x_0, y_0, z_0, tolerance):
    #need an initial h to help the process get started since we need two points for future numerical derivs:
    partialx_0 = (func(x_0,y_0-h,z_0-h)-func(x_0-h,y_0-h,z_0-h))/h
    partialy_0 = (func(x_0-h,y_0,z_0-h)-func(x_0-h,y_0-h,z_0-h))/h
    partialz_0 = (func(x_0-h,y_0,z_0)-func(x_0-h,y_0-h,z_0-h))/h

    x_i = x_0 - gamma*partialx_0
    y_i = y_0 - gamma*partialy_0
    z_i = z_0 - gamma*partialz_0

    step_size = np.sqrt((gamma*partialx_0)**2+(gamma*partialy_0)**2+(gamma*partialz_0)**2)
    i = 1

    # keep track of past steps for plotting 
    func_memory = [func(x_0,y_0,z_0),func(x_i,y_i,z_i)]
    iterations = [0,1]

    while step_size > tolerance: 
        i+=1
        print(i)
        partialx = (func(x_i,y_0,z_0)-func(x_0,y_0,z_0))/(x_i-x_0)
        partialy = (func(x_0,y_i,z_0)-func(x_0,y_0,z_0))/(y_i-y_0)
        partialz = (func(x_0,y_0,z_i)-func(x_0,y_0,z_0))/(z_i-z_0)

        #Take step and get coordinates ready for next numerical derivative.
        x_0 = dc(x_i)
        y_0 = dc(y_i)
        z_0 = dc(z_i)
        x_i += - gamma * partialx
        y_i += - gamma * partialy
        z_i += - gamma * partialz

        step_size = np.sqrt((gamma*partialx)**2+(gamma*partialy)**2+(gamma*partialz)**2)
        func_memory.append(func(x_i,y_i,z_i))
        iterations.append(i)

    x_final = x_i
    y_final = y_i
    z_final = z_i

    return x_final, y_final, z_final, func_memory, iterations

phi_, M_, a_ , chi_arr1, iter_arr1 = grad_descent_3D(chi_squared, 1e-4, 0.01, -3.2, 11.5, -0.5, 1e-5)
phi_2, M_2, a_2 , chi_arr2, iter_arr2 = grad_descent_3D(chi_squared, 1e-4, 0.01, -2.0, 10, -1.5, 1e-5)
phi_3, M_3, a_3 , chi_arr3, iter_arr3 = grad_descent_3D(chi_squared, 1e-4, 0.01, -3.0, 9, -0.8, 1e-5)

print(iter_arr3)
print(chi_arr3)

plt.plot(iter_arr1,chi_arr1, label='-3.2, 11.5, -0.5')
plt.plot(iter_arr2,chi_arr2, label='-2.0, 10, -1.5')
plt.plot(iter_arr3,chi_arr3, label='-3.0, 9, -0.8')
plt.xlabel('Iterations', fontsize=20)
plt.ylabel('Chi^2', fontsize=20)
plt.title('Grad Descent for Different Starting Values', fontsize=24)
plt.xscale('log')
plt.yscale('linear')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
plt.grid(True)
plt.show()



mvals = np.linspace(9.5,12,100)
n_vals = [n(i, phi_, M_ , a_) for i in mvals]
n_vals_i = [n(i, -3.2, 11.5, -0.5) for i in mvals]
print("Chi^2 Value of Fit = ", chi_squared(-2.607, 11.0, -1.046))

plt.errorbar(masses,n_data, yerr=sigma, fmt = 'o', ecolor='black', capsize=5)
plt.plot(mvals,n_vals_i, linestyle = ':', color = 'blue', label='Initial Guess')
plt.plot(mvals,n_vals, color = 'red', label = 'Grad Descent Fit')
plt.xscale('linear') # Set x-axis to logarithmic scale
plt.yscale('log') # Set y-axis to logarithmic scale (optional)
plt.ylabel('n(M_gal) [1/dex/Volume]', fontsize=20)
plt.xlabel('log(M_gal) [dex]', fontsize=20)
plt.title('Chi^2 Fit for Schecter Function', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend()
plt.grid(True)
plt.show()

