# Experiment No. 8
# Customer Churn Prediction using Decision Tree Classification

#NAME - PALLAV PANKAJ
#ROLL - 40

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

# Download Telco Customer Churn dataset and place it
# in the same folder as this Python file.
df = pd.read_csv("Telco_Customer_Churn.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

#NAME - PALLAV PANKAJ
#ROLL - 40
# ---------------------------------------------------------
# 2. Data Preprocessing
# ---------------------------------------------------------

# customerID is not useful for prediction
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# TotalCharges contains some blank values
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

# Fill missing numerical values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical columns
label_encoders = {}

for column in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

print("\nAfter preprocessing:")
print(df.head())

# ---------------------------------------------------------
# 3. Separate Features and Target
# ---------------------------------------------------------

X = df.drop("Churn", axis=1)
y = df["Churn"]

#NAME - PALLAV PANKAJ
#ROLL - 40
# ---------------------------------------------------------
# 4. Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ---------------------------------------------------------
# 5. Train Decision Tree Classifier
# ---------------------------------------------------------

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. Prediction
# ---------------------------------------------------------

y_pred = model.predict(X_test)

#NAME - PALLAV PANKAJ
#ROLL - 40
# ---------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0
))

# ---------------------------------------------------------
# 8. Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

#NAME - PALLAV PANKAJ
#ROLL - 40
# ---------------------------------------------------------
# 9. Visualize Decision Tree
# ---------------------------------------------------------

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree for Customer Churn Prediction")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10. Feature Importance
# ---------------------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance)

plt.figure(figsize=(10, 6))

importance.head(10).sort_values().plot(
    kind="barh"
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features Influencing Customer Churn")
plt.tight_layout()
plt.show()


#NAME - PALLAV PANKAJ
#ROLL - 40

# QUESTIONS:
# 1. What is classification, and how does it differ from regression?

# Classification is a supervised machine-learning technique used to predict a categorical class or label.

# For example:

# Customer → Churn / No Churn
# Email → Spam / Not Spam
# Disease → Positive / Negative

# Regression predicts a continuous numerical value.

# Example:

# House → ₹50,00,000
# Sales → ₹2,50,000
# Temperature → 32.5°C

# Therefore:

# Classification	Regression
# Predicts categories	Predicts numerical values
# Output is discrete	Output is continuous
# Example: Churn/No Churn	Example: Sales prediction
# 2. Explain the working principle of the Decision Tree algorithm.

# A Decision Tree works by repeatedly dividing the dataset into smaller groups using feature-based conditions.

# For example:

#               Tenure < 12?
#                /       \
#              Yes        No
#              /           \
#       Contract?        No Churn
#        /    \
#      Yes    No
#      /       \
#  No Churn   Churn

# The algorithm selects the feature and threshold that provide the best separation of classes.

# The splitting can be based on Gini Index or Entropy, as specified in the experiment.

# 3. Differentiate between Gini Index and Entropy.
# Gini Index	Entropy
# Measures impurity	Measures uncertainty/information disorder
# Generally computationally simpler	Uses logarithmic calculation
# Range is generally 0 to 0.5 for binary classification	Range is 0 to 1 for binary classification
# Lower value indicates purer node	Lower value indicates purer node
# Used by CART	Commonly associated with information gain


# 4. What is a Confusion Matrix? Explain its components.

# A Confusion Matrix is a table used to evaluate a classification model.

# 	Predicted Positive	Predicted Negative
# Actual Positive	TP	FN
# Actual Negative	FP	TN

# Where:

# TP (True Positive): Actual churn and predicted churn.
# TN (True Negative): Actual non-churn and predicted non-churn.
# FP (False Positive): Actual non-churn but predicted churn.
# FN (False Negative): Actual churn but predicted non-churn.
# 5. Define Accuracy, Precision, Recall and F1-Score.
# Accuracy

# Percentage of total predictions that are correct.

# $$ Accuracy=\frac{TP+TN}{TP+TN+FP+FN} $$
# Precision

# Of the customers predicted to churn, how many actually churned?

# $$ Precision=\frac{TP}{TP+FP} $$
# Recall

# Of the customers who actually churned, how many were correctly identified?

# $$ Recall=\frac{TP}{TP+FN} $$
# F1-Score

# Harmonic mean of precision and recall.

# $$ F1=2\times\frac{Precision\times Recall} {Precision+Recall} $$

# These metrics are important because accuracy alone may not adequately represent performance when the classes are imbalanced. The experiment requires all four metrics for evaluation.

# 6. What is overfitting in a Decision Tree? How can it be reduced?

# Overfitting occurs when a Decision Tree becomes excessively complex and learns the training data too closely, including noise.

# It may have:

# Very high training accuracy
#         ↓
# Poor testing accuracy

# Overfitting can be reduced by:

# Limiting max_depth.
# Increasing min_samples_split.
# Increasing min_samples_leaf.
# Pruning the tree.
# Using cross-validation.
# Using ensemble methods such as Random Forest.

# The experiment itself recommends configuring parameters such as maximum tree depth.

# 7. Why is customer churn prediction important for businesses?

# Customer churn prediction helps businesses identify customers who are likely to discontinue their service.

# Early identification allows organizations to:

# Offer discounts or incentives.
# Improve customer service.
# Address customer complaints.
# Provide personalized offers.
# Reduce customer loss.
# Improve customer satisfaction.
# Protect future revenue.

# The lab sheet highlights targeted retention strategies, improved satisfaction and reduced revenue loss as major purposes of churn prediction.

# 8. What are the advantages and limitations of Decision Trees?
# Advantages
# Easy to understand.
# Easy to visualize.
# Requires relatively little preprocessing.
# Can handle categorical and numerical features.
# Provides interpretable decision rules.
# Can identify important features.
# Limitations
# Can easily overfit.
# Small data changes can produce a different tree.
# Very deep trees can become difficult to interpret.
# May perform poorly compared with ensemble methods on some datasets.
# Can be biased toward features with many possible splits.
# 9. Mention three real-world applications of Decision Tree Classification other than customer churn prediction.

# Three applications are:

# Medical diagnosis — predicting whether a patient has a particular disease.
# Credit risk assessment — classifying loan applicants as low or high risk.
# Spam detection — classifying emails as spam or legitimate.

# Other possible applications include fraud detection, employee attrition prediction and sentiment classification.

# 10. How can churn prediction insights help organizations improve customer retention and profitability?

# The model can identify the characteristics of customers who are more likely to churn.

# Organizations can then:

# Identify high-risk customers.
# Provide personalized offers.
# Improve customer support.
# Offer suitable plans or discounts.
# Address problems associated with particular services.
# Focus retention campaigns on high-risk customers.
# Reduce unnecessary marketing expenditure.
# Increase customer lifetime value.

# Thus, churn prediction allows businesses to move from reactive customer retention to proactive retention. The experiment asks students to analyze the factors contributing to churn and recommend strategies based on model predictions.