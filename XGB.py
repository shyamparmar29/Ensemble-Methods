import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()

print(boston.data.shape) #it returns(506, 13). That means there are 506 rows of data with 13 columns
print(boston.feature_names) #feature_names attribute will return the feature names.
print(boston.DESCR) #The description of the dataset is available in the dataset itself. You can take a look at it using .DESCR

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names # To label the names of the columns, use the .columnns attribute of the pandas DataFrame and assign it to boston.feature_names

data.head()

'''You'll notice that there is no column called PRICE in the DataFrame. 
This is because the target column is available in another attribute called boston.target.
Append boston.target to your pandas DataFrame.'''

data['PRICE'] = boston.target

data.info() #to get useful information about the data

'''If we plan to use XGBoost on a dataset which has categorical features you may want to consider applying some
 encoding (like one-hot encoding) to such features before training the model. Also, if you have some missing 
 values such as NA in the dataset you may or may not do a separate treatment for them, because XGBoost is 
 capable of handling missing values internally.'''

X, y = data.iloc[:,:-1],data.iloc[:,-1]   #Separate the target variable and rest of the variables using .iloc to subset the data.

'''Now you will convert the dataset into an optimized data structure called Dmatrix that XGBoost supports
 and gives it acclaimed performance and efficiency gains.  '''

data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

'''The next step is to instantiate an XGBoost regressor object by calling the XGBRegressor() class from the
 XGBoost library with the hyper-parameters passed as arguments. For classification problems, you would have 
 used the XGBClassifier() class.'''
 
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

'''Fit the regressor to the training set and make predictions on the test set using the familiar .fit() and 
.predict() methods. '''

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

#Compute the rmse by invoking the mean_sqaured_error function from sklearn's metrics module.
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


'''k-fold cross validation using XGBoost.You will use different parameters to build a 3-fold cross validation
 model by invoking XGBoost's cv() method and store the results in a cv_results DataFrame. 
 Note that here you are using the Dmatrix object you created before. '''
 
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

cv_results.head()

#Extract and print the final boosting round metric.
print((cv_results["test-rmse-mean"]).tail(1))





