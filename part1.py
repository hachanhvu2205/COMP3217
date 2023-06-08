import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Read the training data
data = pd.read_csv('TrainingDataBinary.csv', sep=',', header=None)

# Split features and result
features = data.iloc[:, :-1]
result = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, result, test_size=0.2, random_state=42)

# MLP Classifier
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
report_mlp = classification_report(y_test, y_pred_mlp)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

# Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
report_rf = classification_report(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
report_knn = classification_report(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Gaussian Naive Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
report_nb = classification_report(y_test, y_pred_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Support Vector Machine Classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
report_svm = classification_report(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
report_dt = classification_report(y_test, y_pred_dt)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Logistic Regression Classifier
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
report_lr = classification_report(y_test, y_pred_lr)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train, y_train)
y_pred_gb = gb_classifier.predict(X_test)
report_gb = classification_report(y_test, y_pred_gb)
accuracy_gb = accuracy_score(y_test, y_pred_gb)

# AdaBoost Classifier
ada_classifier = AdaBoostClassifier()
ada_classifier.fit(X_train, y_train)
y_pred_ada = ada_classifier.predict(X_test)
report_ada = classification_report(y_test, y_pred_ada)
accuracy_ada = accuracy_score(y_test, y_pred_ada)

# Print classification reports and accuracies
print(report_mlp)
print(accuracy_mlp)
print(report_rf)
print(accuracy_rf)
print(report_knn)
print(accuracy_knn)
print(report_nb)
print(accuracy_nb)
print(report_svm)
print(accuracy_svm)
print(report_dt)
print(accuracy_dt)
print(report_lr)
print(accuracy_lr)
print(report_gb)
print(accuracy_gb)
print(report_ada)
print(accuracy_ada)

# Read the testing data
data_predict = pd.read_csv('TestingDataBinary.csv', sep=',', header=None)

# Train the classifier on the entire dataset
rf_classifier.fit(features, result)

# Use the trained random forest classifier to predict the results
predicted_results = rf_classifier.predict(data_predict)

# Assign the predicted results to the respective column in the data
data_predict['PredictedResult'] = predicted_results

# Print the updated data with predicted results
print(data_predict)

# Output the result
data_predict.to_csv('TestingResultsBinary.csv', index=False, header=False)