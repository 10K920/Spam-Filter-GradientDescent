
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE
    wt = w.transpose()
    xTrt = xTr.transpose()
    yTrt = yTr.transpose()
    #print("wt shape in ridge:", wt.shape)
    #print("X_train shape in ridge:", xTr.shape)

    loss = np.sum(np.square(np.dot(wt,xTr)-yTr)) + lambdaa*np.dot(wt,w)
    gradient = 2*(np.dot(xTr,np.dot(xTrt,w))) - 2*np.dot(xTr,yTrt) + 2*lambdaa*w
    
    return loss,gradient
