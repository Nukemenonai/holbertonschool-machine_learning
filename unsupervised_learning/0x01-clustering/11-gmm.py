#!/usr/bin/env python3
"""
kmeans using scikit-learn
"""

import sklearn.mixture


def gmm(X, k):
    """
    calculates GMM from a dataset
    """
    GMM = sklearn.mixture.GaussianMixture(n_components=k)
    GMM.fit(X)

    m = GMM.means_
    S = GMM.covariances_
    pi = GMM.weights_
    clss = GMM.predict(X)
    BIC = GMM.bic(X)

    return pi, m, S, clss, BIC