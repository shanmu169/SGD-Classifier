# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries.

2.Load the Iris dataset.

3.Separate input features (X) and target (y).

4.Split the dataset into training and testing sets.

5.Create an SGD Classifier model.

6.Train the model using fit().

7.Predict species using predict().

8.Calculate accuracy.

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: shanmuga priya K
RegisterNumber: 212225040401 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SGDClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted:", iris.target_names[y_pred])
print("Actual:", iris.target_names[y_test])

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## Output:
![Uploading image.png…]()


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
