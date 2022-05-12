#importing libraries
import numpy as np                  #to work with arrays
import matplotlib.pyplot as plt     #to plot graphs
import pandas as pd                 #to work with datasets and import them

#importing datasets
dataset = pd.read_csv('D:\Machine Learning\Pre Proccessing\Data.csv')   #to import dataset in  variable as data frame
# In Vscode always use absolute path
# The last column of a dataset is called dependent variable
# The other independent columns that determine the last column is called features
x = dataset.iloc[ : , : -1].values  #iloc for index locate x here is all the features
y = dataset.iloc[ : , -1 ].values
# print(x)
# print(y)

#replacing missing data with average in features
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')  #Creating the object
imputer.fit(x[ : ,1:3])
x[ : ,1:3] = imputer.transform(x[ : ,1:3])  #Replaced missing values
#print(x)

#Encoding Categorical Data
#encoding Inependent Variable
from sklearn.compose import ColumnTransformer       #Better than label encoding 
from sklearn.preprocessing import OneHotEncoder     #Type of encoding

ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[0])], remainder='passthrough')
# Creating object the first arguement transformers is a list of tuple with
# the first element being the name for the column transformer
# the second element being the transformer itself in this case OneHotEncoder()
# the third element is the columns to encode
# The second arguement is the remainder by default the constructor returns only encoded columns and drop the rest but by using passthrough arguement it returns all and doesn't changes the rest, we can also specify anither estimator.
x = np.array(ct.fit_transform(x))

# print(x)

#Encoding the dependent variable 
#Label encoding it is safe as long as the output is in binary

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# print(y)

#Splitting the dataset into the training set and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=1)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

#Feature Scaling

from sklearn.preprocessing import StandardScaler   #to apply standardization
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

# print(x_train)
# print(x_test)