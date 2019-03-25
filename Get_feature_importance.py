# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:22:50 2019

@author: Shyam Parmar
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

# Create our test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# build our model
random_forest = RandomForestClassifier(n_estimators=100)

# Train the classifier
random_forest.fit(X_train, y_train)

# Get our features and weights
feature_list = sorted(zip(map(lambda x: round(x, 2), random_forest.feature_importances_), iris.feature_names),
             reverse=True)

# Print them out
print('feature\t\timportance')
print("\n".join(['{}\t\t{}'.format(f,i) for i,f in feature_list]))
print('total_importance\t\t',  sum([i for i,f in feature_list]))