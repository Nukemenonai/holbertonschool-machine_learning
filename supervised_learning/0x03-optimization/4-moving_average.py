#!/usr/bin/env python3
""" contains the moving average function """


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    li = []
    V = 0
    for i in range(len(data)):
    	Vt = (beta * V) + ((1 - beta) * data[i])
	corr = Vt / (1 - (beta ** (i + 1)))
	li.append(corr)
	V = Vt
    return li
