# Logistic Regression in scikit-learn - Lab

## Introduction 

In this lab, you are going to fit a logistic regression model to a dataset concerning heart disease. Whether or not a patient has heart disease is indicated in the column labeled `'target'`. 1 is for positive for heart disease while 0 indicates no heart disease.

## Objectives

In this lab you will: 

- Fit a logistic regression model using scikit-learn 


## Let's get started!

Run the following cells that import the necessary functions and import the dataset: 


```python
# Import necessary functions
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
```


```python
# Import data
df = pd.read_csv('heart.csv')
df.head()
```

## Define appropriate `X` and `y` 

Recall the dataset contains information about whether or not a patient has heart disease and is indicated in the column labeled `'target'`. With that, define appropriate `X` (predictors) and `y` (target) in order to model whether or not a patient has heart disease.


```python
# Split the data into target and predictors
y = df["target"]
X = df.drop("target", axis = 1)
```

## Normalize the data 

Normalize the data (`X`) prior to fitting the model. 


```python
# normalize (subract mean and divide by std)
def normalize(feature):
    return (feature - feature.mean()) / feature.std()

X = X.apply(normalize)
X.head()
```

## Train- test split 

- Split the data into training and test sets 
- Assign 25% to the test set 
- Set the `random_state` to 0 

N.B. To avoid possible data leakage, it is best to split the data first, and then normalize.


```python
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```


## Fit a model

- Instantiate `LogisticRegression`
  - Make sure you don't include the intercept  
  - set `C` to a very large number such as `1e12` 
  - Use the `'liblinear'` solver 
- Fit the model to the training data 


```python
# Instantiate the model
logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')

# Fit the model
logreg.fit(X_train, y_train)
```

## Predict
Generate predictions for the training and test sets. 


```python
# Generate predictions
y_hat_train = logreg.predict(X_train)
y_hat_test = logreg.predict(X_test)
```

## How many times was the classifier correct on the training set?


```python
# Your code here
results = 0
for idx, y in enumerate(y_hat_train):
    if y_hat_train[idx] == list(y_train)[idx]:
        results += 1
        
print(f"The classifier was correct {results} times on the training set out of {len(y_train)}")
```

## How many times was the classifier correct on the test set?


```python
# Your code here

results_test = 0
for idx, y in enumerate(y_hat_test):
    if y_hat_test[idx] == list(y_test)[idx]:
        results_test += 1
        
print(f"The classifier was correct {results_test} times on the test set out of {len(y_test)}")
```

## Analysis
Describe how well you think this initial model is performing based on the training and test performance. Within your description, make note of how you evaluated performance as compared to your previous work with regression.

#### Your analysis here
The model is 82.89% accurate on the test based on the simple calculations above, and 84.58% accurate on the training dataset.

## Summary

In this lab, you practiced a standard data science pipeline: importing data, split it into training and test sets, and fit a logistic regression model. In the upcoming labs and lessons, you'll continue to investigate how to analyze and tune these models for various scenarios.
