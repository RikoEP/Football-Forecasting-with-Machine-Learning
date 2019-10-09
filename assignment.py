# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel('football.xlsx')
X = dataset.iloc[:, :18].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1])
labelencoder_y = LabelEncoder()
X[:, 17] = labelencoder_y.fit_transform(X[:, 17])
onehotencoder = OneHotEncoder(categorical_features = [17])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train = X[:27, 2:]
X_test = X[28:, 2:]
y_train = X[:27, :2]
y_test = X[28:, :2]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

classifier1 = OneVsRestClassifier(LogisticRegression(random_state = 0))
classifier2 = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2))
classifier3 = OneVsRestClassifier(SVC(kernel = 'rbf', random_state = 0))
classifier4 = OneVsRestClassifier(GaussianNB())
classifier5 = OneVsRestClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 0))
classifier6 = OneVsRestClassifier(RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0))

classifier1.fit(X_train, y_train)
classifier2.fit(X_train, y_train)
classifier3.fit(X_train, y_train)
classifier4.fit(X_train, y_train)
classifier5.fit(X_train, y_train)
classifier6.fit(X_train, y_train)

m = 0
l = 0
d = 0

y_pred = classifier1.predict(X_test)
if y_pred[0][0] == 0 and y_pred[0][1] == 1:
    m += 1
if y_pred[0][0] == 1 and y_pred[0][1] == 0:
    l += 1
if y_pred[0][0] == 0 and y_pred[0][1] == 0:
    d += 1
    
y_pred = classifier2.predict(X_test)
if y_pred[0][0] == 0 and y_pred[0][1] == 1:
    m += 1
if y_pred[0][0] == 1 and y_pred[0][1] == 0:
    l += 1
if y_pred[0][0] == 0 and y_pred[0][1] == 0:
    d += 1
    
y_pred = classifier3.predict(X_test)
if y_pred[0][0] == 0 and y_pred[0][1] == 1:
    m += 1
if y_pred[0][0] == 1 and y_pred[0][1] == 0:
    l += 1
if y_pred[0][0] == 0 and y_pred[0][1] == 0:
    d += 1
    
y_pred = classifier4.predict(X_test)
if y_pred[0][0] == 0 and y_pred[0][1] == 1:
    m += 1
if y_pred[0][0] == 1 and y_pred[0][1] == 0:
    l += 1
if y_pred[0][0] == 0 and y_pred[0][1] == 0:
    d += 1
    
y_pred = classifier5.predict(X_test)
if y_pred[0][0] == 0 and y_pred[0][1] == 1:
    m += 1
if y_pred[0][0] == 1 and y_pred[0][1] == 0:
    l += 1
if y_pred[0][0] == 0 and y_pred[0][1] == 0:
    d += 1
    
y_pred = classifier6.predict(X_test)
if y_pred[0][0] == 0 and y_pred[0][1] == 1:
    m += 1
if y_pred[0][0] == 1 and y_pred[0][1] == 0:
    l += 1
if y_pred[0][0] == 0 and y_pred[0][1] == 0:
    d += 1

print('Presentase MUN menang:  %.2f'%((m / (m + l + d)) * 100))
print('Presentase Draw:  %.2f'%((d / (m + l + d)) * 100))
print('Presentase LIV menang:  %.2f'%((l / (m + l + d)) * 100))
