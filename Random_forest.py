# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:30:34 2019

@author: Shyam Parmar
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.io import arff
import pandas as pd

data = arff.loadarff('Autism-Adolescent-Data.arff')
df = pd.DataFrame(data[0])
list(df)
df.head()

#Let’s begin with lowercasing and one-hot encoding the categorical variables so that we can turn the categorical 
#variables to numeric. Let’s make the following changes:

#Lowercase all the text
#All ‘yes’ = 1
#All ‘no’ = 0
#Female = 1
#Male = 0

df = df.apply(lambda x: x.astype(str).str.lower())
df = df.replace('yes', 1)
df = df.replace('no', 0)
df = df.replace('f', 1)
df = df.replace('m', 0)

#Subset Data. Our new dataset should only have the variables that we will be using to build the model.
xVar = list(df.loc[:,'A1_Score':'A10_Score']) + ['gender'] + ['jundice'] + ['austim']
yVar = df.iloc[:,20]
df2 = df[xVar]

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df2, yVar, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

#Build a random forest classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)

clf.fit(X_train, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)

#Predict. Our model is only as good as its predictions, so let’s use it to predict Autism in the test set.
preds = clf.predict(X_test)
print('Prediction : ', preds)

#Check accuracy of model
pd.crosstab(y_test, preds, rownames=['Actual Result'], colnames=['Predicted Result'])
#As we can see, the model did pretty well! It only classified one observation incorrectly.

#Check feature importance.As a final step,let’s look at the importance of all the features in this dataset.
list(zip(X_train, clf.feature_importances_))