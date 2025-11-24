# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Explore the Dataset
2. Preprocess the Data
3. Split the Data and Train the Model
4. Make Predictions and Evaluate the Model


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: LOGU R
RegisterNumber:  212224230141
*/
```
```Python
import pandas as pd
df=pd.read_csv("Salary.csv")
data = pd.DataFrame(df)
print(data)
print(df.head())

print(df.info())

print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
print(df.head())

x=df[["Position","Level"]]
y=df["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("LOGU")
print("Reg No: 212224230141")
print(y_pred)

from sklearn import metrics

mse=metrics.mean_squared_error(y_test,y_pred)
print(mse)

import numpy as np

rmse=np.sqrt(mse)
print(rmse)

r2=metrics.r2_score(y_test,y_pred)
print(r2)

dt.predict(pd.DataFrame([[5, 6]], columns=["Position", "Level"]))

```

## Output:
### DataFrame:>
<img width="364" height="245" alt="image" src="https://github.com/user-attachments/assets/30125884-a2de-4dea-ab5e-90b016aacad6" />

<img width="355" height="130" alt="image" src="https://github.com/user-attachments/assets/410214aa-8dbb-4d91-938e-a414c18a6d5e" />

<img width="455" height="321" alt="image" src="https://github.com/user-attachments/assets/4271ab37-4b24-4edd-a4bd-becdf8a3e51f" />

<img width="319" height="126" alt="image" src="https://github.com/user-attachments/assets/1b910e12-0332-4609-bdaf-9b45be7d30f4" />

<img width="304" height="44" alt="image" src="https://github.com/user-attachments/assets/937ad410-5f46-448a-93b5-8041c669fc58" />

### MSE , RMSE, R2_SCORE:

<img width="206" height="64" alt="image" src="https://github.com/user-attachments/assets/b798b5bb-9bd0-4938-ba57-eabb04fefa8b" />

### predict value : 

<img width="111" height="33" alt="image" src="https://github.com/user-attachments/assets/569c0b2e-56b5-492c-bfc0-0e5d2e801fd0" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
