#6.11 on the overrelaxation method in Newman

import numpy as np


def f(x,c):
    return 1-np.exp(-c*x)

#Part A
def relax_method(x_guess, func, c, tolerance):
    err = 1 #initialize
    i = 0
    while abs(err)>tolerance:
        i+=1
        print(x_guess)
        x_new = func(x_guess,c)
        err = (x_guess-x_new)/(1-(1/(c*np.exp(c*x_guess))))
        x_guess = x_new
    return print('[Relaxation] Converged at x = ', x_guess,' after ', i, ' iterations.')

relax_method(5,f,2,1e-6) #Calls the function to answer part A

#Part B
def overrelax_method(x_guess, func, w, c, tolerance):
    err = 1 #initialize
    i = 0
    while abs(err)>tolerance:
        i+=1
        print(x_guess)
        x_new = (1+w)*func(x_guess,c) - w*x_guess
        err = (x_guess-x_new)/(1-1/((1+w)*c*np.exp(c*x_guess)-w))
        x_guess = x_new
    return print('[Overrelaxation] Converged at x = ', x_guess,' after ', i, ' iterations.')

overrelax_method(5, f, 0.0508, 2, 1e-6) #Calls the function to answer part B, 

#After Trial and error, I found that a value of w = 0.0508 causes it to converge very quickly

