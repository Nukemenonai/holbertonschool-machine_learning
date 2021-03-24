#!/usr/bin/env python3
""" PCA with var """

import numpy as np


def pca(X, var=0.95):
    """ 
    performs PCA on a dataset
    X: numpy.ndarray 
        n is the number of data points
        d number of dimensions in each point
    var: fraction of the variance that PCA should maintain 
    returns: the weight matrix 
    """

    # singulr value decomposition 
    U, S, Vh = np.linalg.svd(X)

    # get the threshold by multipling the last item 
    # of the cumulative sum of singular values by the variance fragment
    c_s_arr = np.cumsum(S)
    threshold = c_s_arr[-1] * var

    mask = np.where(c_s_arr < threshold) 
    r = len(c_s_arr[mask])

    W = Vh.T
    # truncate to the first r values  
    Wr = W[:, :r+1]
    return Wr 

