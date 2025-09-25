#Problem 6.13 From Newman. Implementing the Binary Search Algorithm to find a zero of the function

import numpy as np

def f(x):
    return 5*np.exp(-x)+x-5

def BinarySearch(f,x_low,x_high, tolerance):
    if f(x_low)*f(x_high) < 0: #means that we picked two points that have different signs under function evaluation
        err = x_high-x_low
        i=0
        while err > tolerance:
            i+=1
            x_mid = (x_high+x_low)/2
            print(x_mid)
            if f(x_low)*f(x_mid) < 0:
                x_high = x_mid
            if f(x_high)*f(x_mid) < 0:
                x_low = x_mid
            if f(x_mid) == 0:
                break
            err = (x_high-x_low)/2
        return print('The function has a zero at approximately x = ', (x_high+x_low)/2, '. [Converged after ', i,' iterations]')
    else:
        return print("ERROR: Both points have the same sign, please pick another pair.")

BinarySearch(f, -1.1, 1, 10e-6)
BinarySearch(f, 3, 7, 10e-6)
BinarySearch(f, -1, 7, 10e-6)