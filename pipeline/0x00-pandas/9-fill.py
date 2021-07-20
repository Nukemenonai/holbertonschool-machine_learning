#!/usr/bin/env python3
""" fill missing data points """ 

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

#removing column weighted price 
df.drop('Weighted_Price', axis=1, inplace=True)
# missing values in Close should be set to the previous row value
df["Close"].fillna(method="ffill", inplace=True)
# missing values in High, Low, Open should be set to the same row’s Close value
df["High"].fillna(value=df.Close.shift(1, axis=0), inplace=True)
df["Low"].fillna(value=df.Close.shift(1, axis=0), inplace=True)
df["Open"].fillna(value=df.Close.shift(1, axis=0), inplace=True)
# missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df["Volume_(BTC)"].fillna(0, inplace=True)
df["Volume_(Currency)"].fillna(0, inplace=True)


print(df.head())
print(df.tail())