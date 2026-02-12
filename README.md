# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Algorithm

1. Import all necessary packages and dataset that you need to implement Logistic Regression.

2. Copy the actual dataset and remove fields which are unnecessary.

3. Then select dependent variable and independent variable from the dataset.

4. And perform Logistic Regression.

5. print the values of confusion matrix, accuracy, Classification report to find whether the student is placed or not.

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mahith M
RegisterNumber: 25004610  


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("Placement_Data.csv")


drops = ['sl_no', 'salary']
data = data.drop([c for c in drops if c in data.columns], axis=1)

le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

X = data.drop('status', axis=1)
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

## Output:
<img width="1401" height="411" alt="Screenshot 2026-02-12 112735" src="https://github.com/user-attachments/assets/e2608a6b-85df-4ce0-8cc1-76c433e4a2d8" />
<img width="1406" height="781" alt="Screenshot 2026-02-12 112752" src="https://github.com/user-attachments/assets/dd70b911-5fbd-4803-8cb7-394bed65c89d" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
