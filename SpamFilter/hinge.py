from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w


    # YOUR CODE HERE
    wt = w.transpose()
    xTrt = xTr.transpose()
    yTrt = yTr.transpose()
    
    loss = np.sum(np.maximum(1-yTr*np.dot(wt,xTr), 0)) + lambdaa*np.dot(wt,w)
    gradient = -(np.dot((np.maximum(1-yTr*np.dot(wt,xTr), 0)>0)*yTr,xTrt)).transpose() + 2*lambdaa*w
  
    return loss,gradient
