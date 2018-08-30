"""
Created on Mon Aug 20 16:37:21 2018

@author: vniranja
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset and getting the X and Y
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset[['Level']].values
y = dataset['Salary'].values

plt.scatter(X,y)
plt.show()



# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin= lin_reg.predict(X) 

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
poly_X = poly_reg.fit_transform(X) #Fitting the value of X in polynomial fashion
poly_reg.fit(poly_X, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(poly_X, y)
y_pred_poly = lin_reg_2.predict(poly_X) 

# Visualising the Linear Regression results
plt.scatter(X, y)
plt.plot(X, y_pred_lin)
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y)
plt.plot(X, y_pred_poly)
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#print(regressor.coef_)
from sklearn.metrics import mean_squared_error
from math import sqrt
rms_poly = sqrt(mean_squared_error(y_pred_poly, y))
rms_lin = sqrt(mean_squared_error(y_pred_lin, y))




# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))








