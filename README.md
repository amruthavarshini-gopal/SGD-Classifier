# SGD-Classifier

## Aim:

To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import Necessary Libraries and Load iris Data set.

2.Create a DataFrame from the Dataset.

3.Add Target Labels to the DataFrame.

4.Split Data into Features (X) and Target (y).

5.Split Data into Training and Testing Sets.

6.Initialize the SGDClassifier Model.

7.Train the Model on Training Data.

8.Make Predictions on Test Data.

9.Evaluate Accuracy of Predictions.

10.Generate and Display Confusion Matrix.

11.Generate and Display Classification Report.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Amruthavarshini Gopal
RegisterNumber: 212223230013 
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df.head())
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```

## Output:

![Screenshot 2025-04-28 154943](https://github.com/user-attachments/assets/bf688a8e-e85d-4698-b517-77d31c9d7b2c)

## Result:

Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
