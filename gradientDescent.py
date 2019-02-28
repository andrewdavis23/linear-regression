# gradient descent function for linear regression
# the derivative of the cost function

import numpy as np

def gradientDescent(x,y,theta,alpha,num_iters):
    m = y.size
       
    i = 0
    while i < num_iters:
        theta = theta - (((alpha/m) * x.T) @ (x @ theta - y))
        i += 1
    
    
    return theta