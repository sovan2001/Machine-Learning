
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\Machine Learning\Regression\Multiple Linear Regression\Startups.csv')
x = dataset.iloc[ : , :-1].values
y = dataset.iloc[ : , -1].values
#print(y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
#print(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Backward elimination (Optional)
x = x[:,1:]
import statsmodels.api as sm
x = np.append(arr = np.ones((50,1),dtype= int),values = x, axis = 1)

#Step 2 Create Optimal Array
x_opt = np.array(x[:,[0,1,2,3,4,5]], dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())
index = -1
for i in range(6):
    maxi = -1
    num_columns = np.shape(x_opt)[1]
    for i in range(num_columns):
        if regressor_OLS.pvalues[i] > maxi:
            index = i
            maxi = regressor_OLS.pvalues[i]
    if maxi > 0.05:
        x_opt = np.delete(x_opt,index,1)
        regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
    elif maxi < 0.05:
        break
np.set_printoptions(precision = 2)
print(x_opt)
 