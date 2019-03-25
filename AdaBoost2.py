# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 01:07:40 2019

@author: Shyam Parmar
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.datasets import load_iris
from sklearn.svm import SVC                         # Import Support Vector Classifier
from sklearn import metrics                        #Import scikit-learn metrics module for accuracy calculation


iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4) # 60% training and 40% test

# Create adaboost classifer object
svc = SVC(probability=True, kernel='linear')
abc = AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)

# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))