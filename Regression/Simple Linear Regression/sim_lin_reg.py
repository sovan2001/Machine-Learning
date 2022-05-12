#importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#importing the dataset
dataset = pd.read_csv('D:\Machine Learning\Regression\Simple Linear Regression\Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test results
y_pred = regressor.predict(x_test)

#Plotting the graph for training set
plt.scatter(x_train, y_train, color = 'red')    #Scattering The Points
plt.plot(x_train, regressor.predict(x_train),color = 'blue')    #Plotting The Regression Line
plt.title('Salary vs Years Of Experience (Training Set)')   #Title Of graph
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')                                        
# plt.show()                                                  #To Display the graph

#Plotting The Graph For Test Set
plt.scatter(x_test, y_test, color = 'red')    #Scattering The Points
plt.plot(x_train, regressor.predict(x_train),color = 'blue')    #Plotting The Regression Line
plt.title('Salary vs Years Of Experience (Test Set)')   #Title Of graph
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')                                        
# plt.show()

#Plotting The Graph For All Values
plt.scatter(x, y, color = 'red')    #Scattering The Points
plt.plot(x_train, regressor.predict(x_train),color = 'blue')    #Plotting The Regression Line
plt.title('Salary vs Years Of Experience (Full Set)')   #Title Of graph
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')                                        
plt.show()

#To find the individual value of y in respect to x
print(regressor.predict([[12]]))

#To Find slope or coefficent(m) and intercept(c)
print(regressor.coef_)
print(regressor.intercept_)
