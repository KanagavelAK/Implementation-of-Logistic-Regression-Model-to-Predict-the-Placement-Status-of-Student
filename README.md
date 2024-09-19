# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.

2. Print the placement data and salary data.

3. Find the null and duplicate values.

4. Using logistic regression find the predicted values of accuracy , confusion matrices.

5.  Display the results.


## Program:
```py
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Kanagavel A K
RegisterNumber: 212223230096
import pandas as pd
data1=pd.read_csv('Placement_Data.csv')
data1.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(solver ="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
### op1:
![image](https://github.com/user-attachments/assets/71453f77-4b5f-495f-9436-9cd0769f5d31)
### op2:
![image](https://github.com/user-attachments/assets/0ecc2a49-9112-480d-a1b5-969672e84aa1)
### op3:
![image](https://github.com/user-attachments/assets/85237170-99eb-4cb2-8bcd-326d65c99e77)
### op4:
![image](https://github.com/user-attachments/assets/15bbe57f-6248-4e6d-940d-4279ddde2e5e)
### op5:
![image](https://github.com/user-attachments/assets/3f796858-aea1-46bc-a189-9879d8cad314)
### op6:
![image](https://github.com/user-attachments/assets/7f4fb20e-21ea-4606-9d0b-bbc5ad79b708)
### op7:
![image](https://github.com/user-attachments/assets/efe334fa-2401-48a4-a392-1f8e42490bb1)
### op8:
  ![image](https://github.com/user-attachments/assets/7e77fc00-54b9-43c7-b259-e38e51e8ed68)

### op9:
![image](https://github.com/user-attachments/assets/0e2924aa-cd18-426e-af28-fb5e9efb7df5)

### op10:
![image](https://github.com/user-attachments/assets/ea84e14a-35d4-4739-a2bc-4e67e79a23a2)

### op11:
![image](https://github.com/user-attachments/assets/f2e82f7d-81cb-40c1-9ef8-161a5772d603)
### op12:
![image](https://github.com/user-attachments/assets/4d8184c1-0f65-416b-a357-a64de88aa7a8)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
