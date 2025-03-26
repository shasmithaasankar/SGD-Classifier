# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Shasmithaa Sankar
RegisterNumber: 212224040311

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

 Load the Iris dataset
iris = load_iris()

 Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

 Display the first few rows of the dataset
print(df.head())

 Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

 Train the classifier on the training data
sgd_clf.fit(X_train, y_train)

Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)

Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

 Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()





*/
```

## Output:
![image](https://github.com/user-attachments/assets/e8ee23a3-a842-477f-8699-f7df7275011c)
![image](https://github.com/user-attachments/assets/9fe9c1f0-b933-4a18-929a-e85599a075ec)
![image](https://github.com/user-attachments/assets/c162b316-a71b-4665-abbf-602be1a1a6d5)
![image](https://github.com/user-attachments/assets/8d4f048b-9ecb-4bdc-8b1f-b6e6f5735158)






## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
