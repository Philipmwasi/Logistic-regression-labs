
# Fitting a Logistic Regression Model - Lab

## Introduction

In the last lesson you were given a broad overview of logistic regression. This included an introduction to two separate packages for creating logistic regression models. In this lab, you'll be investigating fitting logistic regressions with `statsmodels`. For your first foray into logistic regression, you are going to attempt to build a model that classifies whether an individual survived the [Titanic](https://www.kaggle.com/c/titanic/data) shipwreck or not (yes, it's a bit morbid).


## Objectives

In this lab you will: 

* Implement logistic regression with `statsmodels` 
* Interpret the statistical results associated with model parameters

## Import the data

Import the data stored in the file `'titanic.csv'` and print the first five rows of the DataFrame to check its contents. 


```python
# Import the data


df = pd.read_csv("titanic.csv")
df.head()

```

## Define independent and target variables

Your target variable is in the column `'Survived'`. A `0` indicates that the passenger didn't survive the shipwreck. Print the total number of people who didn't survive the shipwreck. How many people survived?


```python
# Total number of people who survived/didn't survive

survived = len(df[df["Survived"] == 1])
dead = len(df[df["Survived"] == 0])
print(f'Survived: {survived}\n    Dead: {dead}')
```

Only consider the columns specified in `relevant_columns` when building your model. The next step is to create dummy variables from categorical variables. Remember to drop the first level for each categorical column and make sure all the values are of type `float`: 


```python
# Create dummy variables
relevant_columns = ['Pclass', 'Age', 'SibSp', 'Fare', 'Sex', 'Embarked', 'Survived']
df_relevant = df[relevant_columns]

# Convert categorical columns to type 'category' to ensure proper encoding
for col in df_relevant.select_dtypes(include='object').columns:
    df_relevant[col] = df_relevant[col].astype('category')

dummy_dataframe = pd.get_dummies(df_relevant, drop_first = True, dtype = float)
                                
dummy_dataframe.shape
```

Did you notice above that the DataFrame contains missing values? To keep things simple, simply delete all rows with missing values. 

> NOTE: You can use the [`.dropna()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html) method to do this. 


```python
# Drop missing rows
dummy_dataframe = dummy_dataframe.dropna(axis=0) #dropping by row
dummy_dataframe.shape
```

Finally, assign the independent variables to `X` and the target variable to `y`: 


```python
# Split the data into X and y
y = dummy_dataframe["Survived"]
X = dummy_dataframe.drop("Survived", axis=1)
```

## Fit the model

Now with everything in place, you can build a logistic regression model using `statsmodels` (make sure you create an intercept term as we showed in the previous lesson).  

> Warning: Did you receive an error of the form "LinAlgError: Singular matrix"? This means that `statsmodels` was unable to fit the model due to certain linear algebra computational problems. Specifically, the matrix was not invertible due to not being full rank. In other words, there was a lot of redundant, superfluous data. Try removing some features from the model and running it again.


```python
# Build a logistic regression model using statsmodels
import statsmodels.api as sm

# Add intercept term
X = sm.add_constant(X)

# Fit logistic regression model
logit_model = sm.Logit(y, X)
```

## Analyze results

Generate the summary table for your model. Then, comment on the p-values associated with the various features you chose.


```python
# Summary table
# Obtain the results
logit_result = logit_model.fit()

# Print summary of the model
print(logit_result.summary())
```



### Your comments here
Based on the p-values from the model: The intercept term together with the variables Pclass, Age, SibSp and Sex_male are statistically significant in the model since their p-values are less than 0.05 The variables Fare, Embarked_Q and Embarked_S are not statistically significant since their pvalues are greater than 0.05 which implies that they have little contribution in prediction of the survival.



## Summary 

Well done! In this lab, you practiced using `statsmodels` to build a logistic regression model. You then interpreted the results, building upon your previous stats knowledge, similar to linear regression. Continue on to take a look at building logistic regression models in Scikit-learn!
