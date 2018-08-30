"""
Created on Wed Aug 22 11:51:21 2018

@author: vniranja
"""


# Simple Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset[['YearsExperience']].values
y = dataset['Salary'].values

plt.scatter(X,y)
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)



# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)




# Visualising the Test set results
plt.scatter(X_test, y_test)
plt.plot(X_test, regressor.predict(X_test))
plt.title('Salary vs Experience ')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#print(regressor.coef_)
#from sklearn.metrics import mean_squared_error
#from math import sqrt
#rms = sqrt(mean_squared_error(y_test, y_pred))


#y_pred1 = regressor.predict(3)
#print(y_pred1)

