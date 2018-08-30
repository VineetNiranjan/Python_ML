# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:27:21 2018

@author: vniranja
"""

import numpy as np
import pandas as pd


#pd.options.mode.chained_assignment = None


data = pd.read_csv("titanic.csv")
data.head()


#Fix Gender
data["sex"] = np.where(data["sex"] == "female", 0, 1)


X = data[["pclass", "age", "sex" ]]
y = data["survived"]


# Test train split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test   =  train_test_split(X, y, test_size = 1/5, random_state = 0)


#The Machine learning alogorithm
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier (n_estimators=100)
rf.fit(X_train, y_train)




accuracy = rf.score(X_test, y_test)
print("Accuracy = {}%".format(accuracy * 100))

#rf.predict([[1, 30, 0]])
