# cost function for linear regression
# computes the total cost of the parameters (slope intercept) on the data set

def computeCost(x,y,theta):
    J = ((x @ theta - y).T @ (x @ theta - y)) / (2*y.size)
    return J[0][0]

