# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:45:39 2020

@author: Lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
Y = sc_y.fit_transform(Y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,Y)
Y_pred = regressor.predict(X)
Y=Y.reshape(-1,1)
Y_pred1 = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))


plt.scatter(X,Y, color = 'red')
plt.plot(X, Y_pred , color = 'black')
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position")
plt.ylabel("Salaries")
plt.show()