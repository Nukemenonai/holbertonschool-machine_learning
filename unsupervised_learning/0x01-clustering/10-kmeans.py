#!/usr/bin/env python3
""" sklearn kmmeans""" 

import sklearn.cluster


def kmeans(X, k):

    """performs k mmeans on a data set """
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)

    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
