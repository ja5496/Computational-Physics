# Newman 9.8 - Schrodinger Equation with Crank-Nicholson
import numpy as np
import matplotlib.pyplot as plt

L = 1e-8 #width of square well
x_0 = L/2
sigma = 1e-10
k = 5e10
N = 1000 #number of spacial slices 
hbar = 1
m = 1

def psi_0(L, N):
    a = L/N
    psi = []
    for i in range(1,N+1): #goes from 1 to 1000 instead of 0 to 999
        psi.append(np.exp(-(i*a-x_0)**2/(2*sigma**2)))
    psi = np.array(psi)
    return psi

def A(L, N, dt):
    a = L/N
    A = np.zeros(N,N)
    a_1 = complex(1,dt*hbar/(2*m*a**2))
    a_2 = complex(0,-dt*hbar/(4*m*a**2))
    b_1 = complex(1,-dt*hbar/(2*m*a**2))
    b_2 = complex(0,dt*hbar/(4*m*a**2))

