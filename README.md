# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, drop unnecessary columns, and encode categorical variables.
2.Define the features (X) and target variable (y).
3.Split the data into training and testing sets.
4.Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:SHEIK FAIZAL S
RegisterNumber:24900982
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
```
## Output:
HEAD
![Screenshot 2024-11-20 191127](https://github.com/user-attachments/assets/b7734839-f0d6-4650-96c2-6f9c806e6ede)
COPY
![Screenshot 2024-11-20 191142](https://github.com/user-attachments/assets/485db0c5-c0af-477e-89e3-ea20d68f50af)
FIT TRANSFORM
![Screenshot 2024-11-20 191203](https://github.com/user-attachments/assets/f44a6d2c-0e62-4cbb-a876-e0331df57fde)
LOGISTIC REGRESSION
![Screenshot 2024-11-20 191219](https://github.com/user-attachments/assets/4cf98dbd-7d8f-4013-b328-458844c5bc6c)
ACCURACY SCORE

![Screenshot 2024-11-20 191234](https://github.com/user-attachments/assets/7b555fbd-1a47-4ac7-b7cd-8a3c208522ac)

CLASSIFICATION REPORT

![Screenshot 2024-11-20 191456](https://github.com/user-attachments/assets/eff46a83-7b7b-43a2-bd1a-2c612be4cf08)

PREDICTION

![Screenshot 2024-11-20 191509](https://github.com/user-attachments/assets/95ab44f9-bad8-46ef-a64a-af3a83a38b9a)

![Screenshot 2024-11-20 191524](https://github.com/user-attachments/assets/75751984-3d56-4528-b434-a0d4a865343f)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
