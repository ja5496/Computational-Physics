#Python script for the first problem of Comp Phys HW1 

import numpy as np
import matplotlib.pyplot as plt 
'''
The problem states to differentiate cos(x) and e^x at x = 0.1 and x = 10 using single precision 
with the forward, central, and extrapolated difference methods.

'''
#Part A: Write functions that implement the three methods of numerical differentiation

def forward_diff(f, x, h):
    return ((f(x + h) - f(x)) / h).astype(np.float32)

def central_diff(f, x, h):
    return ((f(x + h) - f(x-h)) / (2*h)).astype(np.float32)

def extrapolated_diff(f, x, h):
    D1 = (f(x + h) - f(x - h)) / (2*h)
    D2 = (f(x + h/2) - f(x - h/2)) / h
    return ((4 * D2 - D1) / 3).astype(np.float32)

x1 = np.float32(0.1)
x2 = np.float32(10.0)

#Compute derivatives of cos(x) and e^x at the h_min's we computed in class that should minimize the error:
f1 = np.cos
f2 = np.exp

h = np.logspace(-8, 1, 100).astype(np.float32) #Range of h values to test

print('forward diff e^x derivative of e^x at x=10', forward_diff(np.exp, x2, np.float32(1.0e-5)))
print('actual derivative of e^x at x=10', np.exp(x2))

#Compute arrays of errors for cosine at both 0.1 and 10.0:
forward_err_f1_x1 = (forward_diff(f1, x1, h) + np.float32(np.sin(x1)))/np.float32(np.sin(x1))
central_err_f1_x1 = (central_diff(f1, x1, h) + np.float32(np.sin(x1)))/np.float32(np.sin(x1))
extrapolated_err_f1_x1 = (extrapolated_diff(f1, x1, h) + np.float32(np.sin(x1)))/np.float32(np.sin(x1))

forward_err_f1_x2 = (forward_diff(f1, x2, h) + np.float32(np.sin(x2)))/np.float32(np.sin(x2))
central_err_f1_x2 = (central_diff(f1, x2, h) + np.float32(np.sin(x2)))/np.float32(np.sin(x2))
extrapolated_err_f1_x2 = (extrapolated_diff(f1, x2, h) + np.float32(np.sin(x2)))/np.float32(np.sin(x2))

#Compute arrays of errors for e^x at both 0.1 and 10.0:
forward_err_f2_x1 = (forward_diff(f2, x1, h) - np.float32(np.exp(x1)))/np.float32(np.exp(x1))
central_err_f2_x1 = (central_diff(f2, x1, h) - np.float32(np.exp(x1)))/np.float32(np.exp(x1))    
extrapolated_err_f2_x1 = (extrapolated_diff(f2, x1, h) - np.float32(np.exp(x1)))/np.float32(np.exp(x1))

forward_err_f2_x2 = (forward_diff(f2, x2, h) - np.float32(np.exp(x2)))/np.float32(np.exp(x2))
central_err_f2_x2 = (central_diff(f2, x2, h) - np.float32(np.exp(x2)))/np.float32(np.exp(x2))
extrapolated_err_f2_x2 = (extrapolated_diff(f2, x2, h) - np.float32(np.exp(x2)))/np.float32(np.exp(x2))

#Part B: Plot the errors for each function at each point using each method on a log-log scale:
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.loglog(h, np.abs(forward_err_f1_x1), label='Forward Diff')
plt.loglog(h, np.abs(central_err_f1_x1), label='Central Diff')
plt.loglog(h, np.abs(extrapolated_err_f1_x1), label='Extrapolated Diff')
plt.title('Errors for cos(x) at x=0.1',fontsize=16, fontweight='bold')
plt.xlabel('Step size (h)',fontsize=14)
plt.ylabel('Relative Error',fontsize=14)
plt.legend()
plt.grid()
plt.subplot(2, 2, 2)
plt.loglog(h, np.abs(forward_err_f1_x2), label='Forward Diff')
plt.loglog(h, np.abs(central_err_f1_x2), label='Central Diff')
plt.loglog(h, np.abs(extrapolated_err_f1_x2), label='Extrapolated Diff')
plt.title('Errors for cos(x) at x=10.0',fontsize=16, fontweight='bold')
plt.xlabel('Step size (h)',fontsize=14)
plt.ylabel('Relative Error',fontsize=14)
plt.legend()
plt.grid()
plt.subplot(2, 2, 3)
plt.loglog(h, np.abs(forward_err_f2_x1), label='Forward Diff')
plt.loglog(h, np.abs(central_err_f2_x1), label='Central Diff')
plt.loglog(h, np.abs(extrapolated_err_f2_x1), label='Extrapolated Diff')
plt.title('Errors for e^x at x=0.1',fontsize=16, fontweight='bold')
plt.xlabel('Step size (h)',fontsize=14)
plt.ylabel('Relative Error',fontsize=14)
plt.legend()
plt.grid()
plt.subplot(2, 2, 4)
plt.loglog(h, np.abs(forward_err_f2_x2), label='Forward Diff')
plt.loglog(h, np.abs(central_err_f2_x2), label='Central Diff')
plt.loglog(h, np.abs(extrapolated_err_f2_x2), label='Extrapolated Diff')
plt.title('Errors for e^x at x=10.0',fontsize=16, fontweight='bold')
plt.xlabel('Step size (h)',fontsize=14)
plt.ylabel('Relative Error',fontsize=14)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

'''

Part C: 
from looking at the plots, we can see that truncation error occurs to the right of the minimum error point,
while rounding error occurs to the left of the minimum error point. This is because the truncation error is defined 
as error due to approximating a mathematical procedure, which becomes more bigger as h increases. Conversely, 
rounding error increases as h decreases because the difference between f(x+h) and f(x) becomes very small, 
leading to cancellation error. Also the more jagged behavior on in the roundoff error regime is due to the  
the lack of significant digits in single precision. (expand more)

'''