# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:27:21 2018

@author: vniranja
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

#The Machine learning alogorithm
from sklearn.linear_model import LogisticRegression

# Test train split
from sklearn.cross_validation import train_test_split


data = pd.read_csv("titanic.csv")
data.head()

#Fix Gender
data["sex"] = np.where(data["sex"] == "female", 0, 1)



X = data[["pclass", "age", "sex" ]]
y = data["survived"]



X_train, X_test, y_train, y_test   =  train_test_split(X, y, test_size = 1/5, random_state = 0)

lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)

y_pred=lr.predict(X_test)

accuracy = lr.score(X_test, y_test)
print("Accuracy = {}%".format(accuracy * 100))

#from sklearn.metrics import confusion_matrix
#cn=confusion_matrix(expected_output_test, y_pred)

