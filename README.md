# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
```
Jhagan B 212220040066 CSE
```
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Jhagan B
RegisterNumber: 212220040066
*/

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
dataset.head()
dataset.tail()

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

#splitting train and test data set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)

#graph plot for traing data
plt.scatter(X_train,Y_train,color = "green")
plt.plot(X_train,reg.predict(X_train),color = "red")
plt.title('Training set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

plt.scatter(X_test,Y_test,color = "blue")
plt.plot(X_test,reg.predict(X_test),color = "silver")
plt.title('Test set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

mse = mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae = mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse = np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
1. df.head()
![WhatsApp Image 2023-09-04 at 11 05 46 PM](https://github.com/jhaganb/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/63654882/2f7a5cf6-f0e9-4d2c-a026-ad1d2f88570b)

   
2. df.tail()
![WhatsApp Image 2023-09-04 at 11 05 46 PM (1)](https://github.com/jhaganb/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/63654882/18749620-4388-4d0c-9d62-72c8f67a9388)

   
3. Array value of X
![WhatsApp Image 2023-09-04 at 11 05 45 PM](https://github.com/jhaganb/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/63654882/e51306f0-f718-42d6-a5dc-973aacc1f226)

   
4. Array value of Y
![WhatsApp Image 2023-09-04 at 11 05 45 PM (1)](https://github.com/jhaganb/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/63654882/9c7b81de-5dbe-4235-9d39-fcd128dad43e)

   
5. Values of Y prediction
![WhatsApp Image 2023-09-04 at 11 05 46 PM (5)](https://github.com/jhaganb/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/63654882/b7a7730b-e141-438a-944a-76787b4a2b7d)

 
6. Values of Y test
![WhatsApp Image 2023-09-04 at 11 05 46 PM (6)](https://github.com/jhaganb/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/63654882/138a76f5-e45e-4329-85b9-0ea19e2dde47)

    
7. Training Set Graph
![WhatsApp Image 2023-09-04 at 11 05 46 PM (4)](https://github.com/jhaganb/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/63654882/e4a7541c-4c8b-4c30-92ad-31fb2a81d61e)

    
8. Test Set Graph
![WhatsApp Image 2023-09-04 at 11 05 46 PM (3)](https://github.com/jhaganb/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/63654882/2c54ea6d-8800-4811-bdf5-0e6881882717)

    
9. Values of MSE, MAE and RMSE
![WhatsApp Image 2023-09-04 at 11 05 46 PM (2)](https://github.com/jhaganb/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/63654882/b6d2e389-c577-4710-bef1-4731a8119849)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
