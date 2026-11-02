# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries.

2.create a dataset.

3.Separate input features (X) and target (y).

4.Split data into training and testing sets.

5.Create an SGDClassifier model.

6.Train the model using fit().

7.Predict using predict().

8.Evaluate accuracy using accuracy_score(). 

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: shanmuga priya K
RegisterNumber: 212225040401 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
              [5, 6], [6, 7], [7, 8], [8, 9]])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = SGDClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## Output:
<img width="980" height="115" alt="image" src="https://github.com/user-attachments/assets/28fcc4c1-71ec-4800-97ca-93b9ffb3e0cb" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
