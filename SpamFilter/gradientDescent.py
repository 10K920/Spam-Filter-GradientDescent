import numpy as np
from numpy import linalg as LA
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent

    # YOUR CODE HERE
    w = w0
    bestLoss = float('inf')
    
    for i in range(maxiter):        
        curLoss, curGrad = func(w)

        if LA.norm(curGrad) < tolerance:
            break

        if curLoss <= bestLoss:
            stepsize = 1.01*stepsize
        else:
            stepsize = 0.5*stepsize

        bestLoss = curLoss
        w = w + (-stepsize*curGrad)
    
    return w, bestLoss
    
    
