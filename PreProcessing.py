"""
Created on Wed Aug 22 11:51:21 2018

@author: vniranja
"""


import pandas
import numpy


data = pandas.read_csv("Diabetes_Data.csv")
array = data.values
X = array[:,0:8]
Y = array[:,8]

#Min Max Scalar
"""
X' = (X - min(x))/(max(x) - min(X))

"""
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])


#Mean Normalize 
"""
X' = (X - avg(x))/(max(x) - min(X))

"""
from sklearn.preprocessing import Normalizer

scalar1 = Normalizer()
reNormalizedX = scalar1.fit_transform(X)


numpy.set_printoptions(precision=3)
print(reNormalizedX[0:5,:])


#Standardization 

"""
X' = (X - avg(x))/σ
"""
from sklearn.preprocessing import StandardScaler

scaler2 = StandardScaler()
rescaledX = scaler2.fit_transform(X)

numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])

#Binarize
from sklearn.preprocessing import Binarizer

binarizer = Binarizer()
binaryX = binarizer.fit_transform(X)

numpy.set_printoptions(precision=3)
print(binaryX[0:5,:])


