#################################
#  Linear Regression Main File  #
#################################
# Data must be in [x,y] format with no headers

import numpy as np                                        # aka MATLAB for poor people

from computeCost import computeCost                       # ML functions
from gradientDescent import gradientDescent
from splitData import splitData

fileName = 'ex1data1.txt'                                                
ratio = 0.66                                              # the portion of data to be used as the training set

splitedData = splitData(fileName,ratio)                   # ML function returns tuple (Training Set, Test Set)

xTest = (splitedData[1])[:,0]                             # all the important info
yTest = (splitedData[1])[:,1]  
y = (splitedData[0])[:,1]            
m = y.size
n = yTest.size
x = (splitedData[0])[:,0]
theta = np.zeros((2,1))
alpha = 0
num_iters = 0

x = x.reshape(m,1)                                        # numpy cancer
y = y.reshape(m,1)
xTest = xTest.reshape(n,1)      
yTest = yTest.reshape(n,1)

x = np.append(np.ones((m,1)),x,axis=1)                    # append a column of ones. Since we're using arrays, we don't have to create a function for the parameterized line
xTest = np.append(np.ones((n,1)),xTest,axis=1)            # computeCost will multiply the parameters by the new x matrices and sum the total cost for all data without a loop

print("Initial parameters: m=",theta[0][0],"b=",theta[1][0],"\nInitial cost =",computeCost(x,y,theta))

alpha = 0.01                                              # alpha = learning rate (size of the step). Value too high -> parameters will diverge and overload the variables.  Value too low -> parameters will take longer to reach minimum cost, will require more iterations
num_iters = 1500                                          # num_iters = number of iterations (the number of steps to take during gradient descent)

theta = gradientDescent(x,y,theta,alpha,num_iters)

print("Minimized cost:",computeCost(x,y,theta))           # the algorithim choose this as the minimum cost for the training set...
print("Final parameters:",theta.T)                        # ...using these parameters
print("Test Set cost:",computeCost(xTest,yTest,theta))    # the cost of predictions on the test set using the parameters learned from the training set









  