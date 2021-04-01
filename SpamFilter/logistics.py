import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):

    # YOUR CODE HERE:
    wt = w.transpose()
    xTrt = xTr.transpose()
    yTrt = yTr.transpose()

    loss = np.sum(np.log(1+np.exp(-yTr*np.dot(wt,xTr))))
    gradient =  np.dot(xTr,((-yTr*np.exp(-yTr*(np.dot(wt,xTr)))/(1+np.exp(-yTr*(np.dot(wt,xTr)))))).transpose())
    
    return loss,gradient
