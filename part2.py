import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Read the training data
data = pd.read_csv('TrainingDataMulti.csv', sep=',', header=None)

# Split features and result
features = data.iloc[:, :-1]  # Select all columns except the last one as features
result = data.iloc[:, -1]  # Select the last column as the result

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, result, test_size=0.2, random_state=42)

# K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
report_knn = classification_report(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Logistic Regression Classifier
logreg_classifier = LogisticRegression()
logreg_classifier.fit(X_train, y_train)
y_pred_logreg = logreg_classifier.predict(X_test)
report_logreg = classification_report(y_test, y_pred_logreg)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
report_dt = classification_report(y_test, y_pred_dt)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
report_rf = classification_report(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Support Vector Machine Classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
report_svm = classification_report(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train, y_train)
y_pred_gb = gb_classifier.predict(X_test)
report_gb = classification_report(y_test, y_pred_gb)
accuracy_gb = accuracy_score(y_test, y_pred_gb)

# Gaussian Naive Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
report_nb = classification_report(y_test, y_pred_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Print the results
print(report_knn)
print("Accuracy:", accuracy_knn)
print()

print(report_logreg)
print("Accuracy:", accuracy_logreg)
print()

print(report_dt)
print("Accuracy:", accuracy_dt)
print()

print(report_rf)
print("Accuracy:", accuracy_rf)
print()

print(report_svm)
print("Accuracy:", accuracy_svm)
print()

print(report_gb)
print("Accuracy:", accuracy_gb)
print()

print(report_nb)
print("Accuracy:", accuracy_nb)
