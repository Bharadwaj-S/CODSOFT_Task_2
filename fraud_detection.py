#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# Paths to the dataset files (save the datasets in the same folder)
train_data_file = "fraudTrain.csv"
test_data_file = "fraudTest.csv"

# Loading the datasets
train_df = pd.read_csv(train_data_file)
test_df = pd.read_csv(test_data_file)

# Dropping non-relevant columns
irrelevant_columns = ['trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 
                      'zip', 'lat', 'long', 'job', 'dob', 'trans_num', 'unix_time', 'merchant']

X_train = train_df.drop(columns=['is_fraud'] + irrelevant_columns)  # Features
y_train = train_df['is_fraud']  # Target variable

X_test = test_df.drop(columns=['is_fraud'] + irrelevant_columns)  # Features for testing
y_test = test_df['is_fraud']  # Target variable for testing

# Identify categorical features
categorical_features = ['category', 'gender']  # Add more if needed
numeric_features = X_train.columns.difference(categorical_features)

# Preprocessing pipeline for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Splitting the training data into training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Apply preprocessing pipeline to training and validation data
X_train_split = preprocessor.fit_transform(X_train_split)
X_val = preprocessor.transform(X_val)
X_test_scaled = preprocessor.transform(X_test)

# Logistic Regression Model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_split, y_train_split)
y_pred_logreg = logreg.predict(X_val)
accuracy_logreg = accuracy_score(y_val, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg:.4f}")

# Decision Tree Model
dtree = DecisionTreeClassifier(max_depth=5)  # Adjust max_depth as needed
dtree.fit(X_train_split, y_train_split)
y_pred_dtree = dtree.predict(X_val)
accuracy_dtree = accuracy_score(y_val, y_pred_dtree)
print(f"Decision Tree Accuracy: {accuracy_dtree:.4f}")

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)  # Adjust n_estimators and max_depth as needed
rf.fit(X_train_split, y_train_split)
y_pred_rf = rf.predict(X_val)
accuracy_rf = accuracy_score(y_val, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# Evaluation on the test set with the best-performing model (Random Forest in this example)
y_test_pred = rf.predict(X_test_scaled)
print("\nRandom Forest Test Set Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# ROC Curve and AUC
y_test_proba = rf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = roc_auc_score(y_test, y_test_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plotting Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
