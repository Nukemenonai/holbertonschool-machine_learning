#!/usr/bin/env python3
"""
agglomerative clustering scipy
"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    performs agglomerative clustering on a dataset:
    """
    hierarchy = scipy.cluster.hierarchy
    linkage_mat = hierarchy.linkage(y=X, method='ward')
    fcluster = hierarchy.fcluster(Z=linkage_mat, t=dist, criterion='distance')
    plt.figure()
    hierarchy.dendrogram(linkage_mat, color_threshold=dist)
    plt.show()
    return fcluster
