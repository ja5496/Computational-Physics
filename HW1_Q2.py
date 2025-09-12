#Python file to solve the numerical integration problem from HW1 Q2
import numpy as np
import matplotlib.pyplot as plt

'''
A) First define integration functions. Use midpoint rule, trapezoidal rule, and Simpson's rule 
to compute the integral of each function from 0 to 1.

'''

def midpoint_rule(f, a, b, N):
    width = (b - a) / N
    integral = 0.0
    for i in range(N):
        x_mid = a + (i + 0.5) * width
        integral += f(x_mid)
    integral *= h
    return integral

def trapezoidal_rule(f, a, b, N):
    width = (b - a) / N
    integral = 0.5 * (f(a) + f(b))
    for i in range(1, N):
        x_i = a + i * width
        integral += f(x_i)
    integral *= width
    return integral

#NOT FINISHED YET, HAVEN'T LEARNED SIMPSON'S RULE