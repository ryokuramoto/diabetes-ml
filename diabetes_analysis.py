"""
This code uses the Pima Indian Diabetes Dataset from the National Institute of Diabetes
and Digestive and Kidney Diseases.
The dataset is available in the public domain under the Creative Commons Zero (CC0) license.
Source: National Institute of Diabetes and Digestive and Kidney Diseases
Original Donor: Vincent Sigillito (vgs@aplcen.apl.jhu.edu), Research Center, RMI Group Leader,
Applied Physics Laboratory, The Johns Hopkins University.
Date Received: 9 May 1990
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# Create the folder "png"
filepath = "./png"
try:
    os.makedirs(filepath)
    print("Folder created successfully")
except FileExistsError:
    print("Folder already exists")
except Exception as e:
    print("An error occurred")

# Diabetes Dataset Preparation -------------------------------------------------

# Read the dataset
df = pd.read_csv('diabetes.csv')

# Print DataFrame info and first few rows
print("\nDataFrame Info:")
df.info()

print("\nDataFrame Values:")
print(df.head())

# Count of duplicated rows
duplicated_rows = df.duplicated().sum()

# Count of zero values per column
zero_values_per_column = (df == 0).sum()

# Printing the results
print(f'Duplicated rows are: \n{duplicated_rows}\n')
print(f'Zero values per column are: \n{zero_values_per_column}\n')

# Identify outliers based on conditions where certain columns have a value of 0
outliers = (
        (df["Glucose"] == 0) |
        (df["BloodPressure"] == 0) |
        (df["SkinThickness"] == 0) |
        (df["Insulin"] == 0) |
        (df["BMI"] == 0)
)

print(f'Number of outliers: {outliers.sum()}\n')

lst = {'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'}
for i in lst:
    df[i] = df[i].replace(0, np.nan)  # Replace 0s with NaN
    df[i] = df[i].fillna(df[i].median())  # Replace NaNs with the median of the column

# Plot pair plot for all data.
def plot_pairplot(df, title):
    plt.figure(figsize=(12, 8))
    sns.pairplot(df, hue='Outcome')
    filename = filepath + "/" + title + ".png"
    plt.savefig(filename)
    plt.close()

plot_pairplot(df, "PairPlot_AllData")

X = df.drop(['Outcome'], axis=1)
y_true = df.Outcome

# Split data set into train and test data.
train_X, test_X, train_y, test_y = train_test_split(X, y_true, test_size=0.2, random_state=42)

# Decision Tree--------------------------------------
print("-----------------Decision Tree-----------------")

# Define the parameter grid for max_depth
param_grid = {'max_depth': np.arange(1, 30)}

# Initialize the Decision Tree Classifier
# Fix random state to reproduce the same result for decision tree node splits
# When multiple features provide the same splitting criterion value (e.g., Gini impurity or entropy),
# the algorithm might randomly select one of the features to split the node
clf = tree.DecisionTreeClassifier(random_state=42)

# Initialize KFold with random_state and shuffle=True
# Fix random state to reproduce the same result for K-Fold cross-validation splits
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kf, scoring='accuracy')

# Fit GridSearchCV to the training data
grid_search.fit(train_X, train_y)


# Define the function to plot palameter effect
def plot_parameter_effect(grid_search, param, model, xlabel, ylabel):
    print(f'\nBest Parameters: {grid_search.best_params_}')
    print(f'Best Cross-Validation Score (Accuracy): {grid_search.best_score_}')

    # Extract mean test scores for each value of the parameter
    mean_test_scores = grid_search.cv_results_['mean_test_score']
    param_values = grid_search.cv_results_[f'param_{param}'].data

    plt.figure(figsize=(10, 6))
    plt.plot(param_values, mean_test_scores, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Effect of " + param + " on Model Performance (" + model + ")")
    plt.grid(True)
    filename = filepath + "/Effect_" + param + "_" + model + ".png"
    plt.savefig(filename)
    plt.close()

# Plot max_depth vs. accuracy
plot_parameter_effect(grid_search, "max_depth", "DecisionTree",
                      "max_depth", "Mean Cross-Validated Accuracy")

# Define the function to evaluate the model and plot confusion matrix
def evaluate_model(grid_search, X, y, data_type, Algorithm):
    print(f"{data_type} data\n{classification_report(y, grid_search.predict(X), digits=3)}")
    conf_mat = confusion_matrix(y, grid_search.predict(X))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    disp.plot(cmap='Blues')
    disp.ax_.set_title(f"Confusion Matrix ({Algorithm}, {data_type} Data)")
    filename = filepath + f"/ConfusionMatrix_{Algorithm}_{data_type}.png"
    plt.savefig(filename)
    plt.close()

# Evaluate the model on training data
evaluate_model(grid_search, train_X, train_y, "Training", "Decision Tree")

# Plot pair plot for the train data
df_train_true = pd.DataFrame(train_X)
df_train_true['Outcome'] = train_y
df_train_pre = pd.DataFrame(train_X)
df_train_pre['Outcome'] = grid_search.predict(train_X)
plot_pairplot(df_train_true, "PairPlot_DecisionTree_TrainingData_True")
plot_pairplot(df_train_pre, "PairPlot_DecisionTree_TrainingData_Predict")

# Evaluate the model on test data
evaluate_model(grid_search, test_X, test_y, "Test", "Decision Tree")

# Plot pair plot for the test data
df_test_true = pd.DataFrame(test_X)
df_test_true['Outcome'] = test_y
df_test_pre = pd.DataFrame(test_X)
df_test_pre['Outcome'] = grid_search.predict(test_X)
plot_pairplot(df_test_true, "PairPlot_DecisionTree_TestData_True")
plot_pairplot(df_test_pre, "PairPlot_DecisionTree_TestData_Predict")

# Plot the decision tree
plt.figure(figsize=(30, 10))
tree.plot_tree(grid_search.best_estimator_, filled=True, fontsize=7)
plt.title("Tree Plot (Decision Tree)")
filename = filepath + "/" + "TreePlot_DecisionTree" + ".png"
plt.tight_layout()
plt.savefig(filename)
plt.close()
# Decision Tree End ----------------------------------

# K Nearest Neighbors--------------------------------------
print("-----------------K Nearest Neighbors-----------------")

# Define the parameter grid
param_grid = {'n_neighbors': range(1, 30)}

# Initialize the K Nearest Neighbors Classifier
clf = KNeighborsClassifier()

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kf, scoring='accuracy')

# Fit GridSearchCV to the training data
grid_search.fit(train_X, train_y)

# Plot n_neighbors vs. accuracy
plot_parameter_effect(grid_search, "n_neighbors", "K Nearest Neighbors",
                      "n_neighbors", "Mean Cross-Validated Accuracy")

# Evaluate the model on training data
evaluate_model(grid_search, train_X, train_y, "Training", "K Nearest Neighbors")

# Plot pair plot for the train data
df_train_pre = pd.DataFrame(train_X)
df_train_pre['Outcome'] = grid_search.predict(train_X)
plot_pairplot(df_train_true, "PairPlot_KNearestNeighbors_TrainingData_True")
plot_pairplot(df_train_pre, "PairPlot_KNearestNeighbors_TrainingData_Predict")

# Evaluate the model on test data
evaluate_model(grid_search, test_X, test_y, "Test", "K Nearest Neighbors")

# Plot pair plot for the test data
df_test_pre = pd.DataFrame(test_X)
df_test_pre['Outcome'] = grid_search.predict(test_X)
plot_pairplot(df_test_true, "PairPlot_KNearestNeighbors_TestData_True")
plot_pairplot(df_test_pre, "PairPlot_KNearestNeighbors_TestData_Predict")
# K Nearest Neighbors End ----------------------------------

# Support Vector Machine--------------------------------------
print("-----------------Support Vector Machine-----------------")

# Define the parameter grid
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.0001, 0.001, 0.01, 0.1]}

# Initialize the Support Vector Machine Classifier
clf = svm.SVC()

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kf, scoring='accuracy')

# Fit GridSearchCV to the training data
grid_search.fit(train_X, train_y)

# Define the function to plot parameter effect heatmap
def plot_parameter_effect_heatmap(grid_search, param1, param2, model):
    print(f'\nBest Parameters: {grid_search.best_params_}')
    print(f'Best Cross-Validation Score (Accuracy): {grid_search.best_score_}')

    # Extract mean test scores for each combination of parameters
    mean_test_scores = grid_search.cv_results_['mean_test_score']
    param1_values = grid_search.cv_results_[f'param_{param1}'].data
    param2_values = grid_search.cv_results_[f'param_{param2}'].data

    # Create a DataFrame for the heatmap
    scores_df = pd.DataFrame({
        param1: param1_values,
        param2: param2_values,
        'Mean Test Score': mean_test_scores
    })

    # Pivot the DataFrame to get a matrix suitable for heatmap
    scores_matrix = scores_df.pivot(index=param2, columns=param1, values='Mean Test Score')

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(scores_matrix, annot=True, fmt=".3f", cmap='coolwarm', linewidths=.5)
    plt.title("Effect of " + param1 + " and " + param2 + " on Model Performance (" + model + ")")
    plt.xlabel(param1)
    plt.ylabel(param2)
    filename = filepath + "/Effect_" + param1 + "_" + param2 + "_" + model + ".png"
    plt.savefig(filename)
    plt.close()

# Plot C vs. gamma vs. accuracy heatmap
plot_parameter_effect_heatmap(grid_search, "C", "gamma", "SupportVectorMachine")

# Evaluate the model on training data
evaluate_model(grid_search, train_X, train_y, "Training", "Support Vector Machine")

# Plot pair plot for the train data
df_train_pre = pd.DataFrame(train_X)
df_train_pre['Outcome'] = grid_search.predict(train_X)
plot_pairplot(df_train_true, "PairPlot_SupportVectorMachine_TrainingData_True")
plot_pairplot(df_train_pre, "PairPlot_SupportVectorMachine_TrainingData_Predict")

# Evaluate the model on test data
evaluate_model(grid_search, test_X, test_y, "Test", "Support Vector Machine")

# Plot pair plot for the test data
df_test_pre = pd.DataFrame(test_X)
df_test_pre['Outcome'] = grid_search.predict(test_X)
plot_pairplot(df_test_true, "PairPlot_SupportVectorMachine_TestData_True")
plot_pairplot(df_test_pre, "PairPlot_SupportVectorMachine_TestData_Predict")
# Support Vector Machine End ----------------------------------



