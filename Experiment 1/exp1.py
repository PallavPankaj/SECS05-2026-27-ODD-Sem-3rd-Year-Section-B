import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("Titanic-Dataset.csv")

print("=" * 60)
print("TITANIC DATA PREPROCESSING")
print("=" * 60)

print("\nFirst 5 Records:")
print(df.head())


# --------------------------------------------------
# 2. DATASET INFORMATION
# --------------------------------------------------

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# --------------------------------------------------
# 3. MISSING VALUES
# --------------------------------------------------

print("\nMissing Values Before Treatment:")
print(df.isnull().sum())

# Age -> Median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked -> Mode
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# Cabin -> Remove
if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

print("\nMissing Values After Treatment:")
print(df.isnull().sum())


# --------------------------------------------------
# 4. REMOVE DUPLICATES
# --------------------------------------------------

print("\nDuplicate Records:",
      df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Duplicates After Removal:",
      df.duplicated().sum())


# --------------------------------------------------
# 5. ENCODE CATEGORICAL VARIABLES
# --------------------------------------------------

# Label Encoding for Sex
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])

# One-Hot Encoding for Embarked
df = pd.get_dummies(
    df,
    columns=["Embarked"],
    drop_first=True
)

# Convert Boolean columns to integers
for col in df.select_dtypes(include="bool").columns:
    df[col] = df[col].astype(int)


# --------------------------------------------------
# 6. OUTLIER DETECTION USING IQR
# --------------------------------------------------

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nFare IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

outliers = df[
    (df["Fare"] < lower_bound) |
    (df["Fare"] > upper_bound)
]

print("Number of Fare Outliers:",
      len(outliers))


# --------------------------------------------------
# 7. OUTLIER TREATMENT
# --------------------------------------------------

df["Fare"] = df["Fare"].clip(
    lower=lower_bound,
    upper=upper_bound
)


# --------------------------------------------------
# 8. FEATURE ENGINEERING
# --------------------------------------------------

df["FamilySize"] = (
    df["SibSp"] +
    df["Parch"] +
    1
)

df["IsAlone"] = np.where(
    df["FamilySize"] == 1,
    1,
    0
)

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=[
        "Child",
        "Teenager",
        "Adult",
        "Middle_Aged",
        "Senior"
    ]
)

df = pd.get_dummies(
    df,
    columns=["AgeGroup"],
    drop_first=True
)

for col in df.select_dtypes(include="bool").columns:
    df[col] = df[col].astype(int)


# --------------------------------------------------
# 9. STANDARDIZATION
# --------------------------------------------------

scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(
    df[["Age", "Fare"]]
)


# --------------------------------------------------
# 10. SAVE CLEANED DATASET
# --------------------------------------------------

df.to_csv(
    "Titanic_Cleaned_Preprocessed.csv",
    index=False
)

print("\nFinal Dataset:")
print(df.head())

print("\nFinal Shape:", df.shape)

print("\nCleaned dataset saved as:")
print("Titanic_Cleaned_Preprocessed.csv")

print("=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)


""" 
  1. Why is data preprocessing considered one of the most important phases in data analytics?

Data preprocessing is important because real-world data is often incomplete, inconsistent, noisy, and contains errors. Before visualization or machine learning, the data must be cleaned and transformed to improve its quality and reliability.

It involves:

Handling missing values
Removing duplicate records
Correcting inconsistent data
Encoding categorical variables
Detecting and treating outliers
Scaling numerical features

Good preprocessing helps produce more reliable analytical results and improves machine-learning model performance.



2. Explain different methods of handling missing values with suitable examples.

Common methods include:

a) Mean

Replace missing numerical values with the mean.

df["Age"].fillna(df["Age"].mean())

Suitable when the data is approximately normally distributed.

b) Median

Replace missing values with the median.

df["Age"].fillna(df["Age"].median())

Useful when numerical data contains outliers.

c) Mode

Replace missing categorical values with the most frequently occurring value.

df["Embarked"].fillna(df["Embarked"].mode()[0])
d) Row Removal

Rows containing missing values can be removed.

df.dropna()

This should be used carefully because too many rows may be lost.

e) Column Removal

A column with an excessive number of missing values can be removed.

df.drop("Cabin", axis=1)

The assignment explicitly lists mean, median, mode, and row/column removal as possible approaches.




3. Differentiate between Label Encoding and One-Hot Encoding.
Label Encoding	One-Hot Encoding
Converts categories into numerical labels	Creates separate binary columns
Usually produces one column	Produces multiple columns
Example: Male = 1, Female = 0	Example: Male = [1,0], Female = [0,1]
Suitable for ordinal categories in many cases	Useful for nominal categories
Easy to implement	Can increase the number of features

Example:

Label Encoding:

Male → 1
Female → 0

One-Hot Encoding:

Embarked_S  Embarked_Q
1           0
0           1

Both are mentioned in the assignment as methods for converting categorical variables into numerical form.




4. What are outliers? How can they affect analytical results?

Outliers are observations that are unusually far away from the majority of values in a dataset.

For example, if most passenger fares are between ₹10 and ₹100 but one passenger has a fare of ₹500, that value may be an outlier.

Outliers can:

Distort the mean
Increase variance
Affect statistical analysis
Influence correlations
Affect machine-learning models
Produce misleading visualizations

One common method for detecting them is the Interquartile Range (IQR) method, as required in this experiment.




5. Explain the difference between normalization and standardization.
Normalization

Normalization scales values to a fixed range, commonly 0 to 1.

Formula:

$$ X_{normalized} = \frac{X-X_{min}}{X_{max}-X_{min}} $$
Standardization

Standardization transforms values so that they have approximately:

Mean = 0
Standard deviation = 1

Formula:

$$ X_{standardized} = \frac{X-\mu}{\sigma} $$
Difference
Normalization	Standardization
Usually scales to 0–1	Centers around 0
Uses minimum and maximum	Uses mean and standard deviation
Sensitive to extreme min/max values	Generally less dependent on range
Useful when a bounded range is desired	Commonly used for many ML algorithms

The assignment requires numerical attributes to be normalized or standardized to bring them to a common scale.




6. Why should duplicate records be removed before analysis?

Duplicate records can cause the same observation to be counted multiple times.

This can:

Distort statistical calculations
Produce incorrect frequencies
Bias analytical results
Affect machine-learning models
Give incorrect conclusions

Therefore, duplicate records should be detected and removed to maintain data consistency.

Python:

df.drop_duplicates(inplace=True)




7. What is feature engineering? Give two practical examples.

Feature engineering is the process of creating new meaningful attributes from existing data to improve analysis or machine-learning performance.

Example 1 — Family Size

In Titanic:

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

This combines the number of siblings/spouses and parents/children with the passenger to determine family size.

Example 2 — Age Group

Passengers can be divided into categories:

Child
Teenager
Adult
Middle_Aged
Senior

This converts continuous age information into meaningful groups.

The assignment specifically gives Family Size and Age Group as examples of feature engineering.




8. Which preprocessing techniques would you apply to the IBM HR Employee Attrition dataset and why?

For the IBM HR Employee Attrition dataset, I would apply:

Missing-value handling — identify and appropriately treat missing employee information.
Duplicate removal — prevent employees from being counted multiple times.
Categorical encoding — convert attributes such as department, job role, gender, and marital status into numerical form.
Outlier detection — identify unusual values in numerical attributes such as income or age.
Normalization/standardization — bring numerical attributes to a comparable scale.
Feature engineering — create useful attributes such as age groups or income categories.
Data validation — check inconsistent or invalid values.

These techniques directly correspond to the preprocessing methods specified in the experiment.




9. How does poor-quality data affect machine learning model performance?

Poor-quality data can significantly reduce model performance.

For example:

Missing values can prevent algorithms from learning properly.
Duplicate records can introduce bias.
Incorrect values can lead to incorrect patterns.
Outliers can distort the model.
Unencoded categorical variables cannot be directly processed by many algorithms.
Features with very different scales can cause some algorithms to give excessive importance to certain features.

Therefore, preprocessing improves the quality and reliability of the data before machine learning.




10. Name any three Python libraries commonly used for data preprocessing.

Three commonly used Python libraries are:

Pandas — data manipulation and cleaning
NumPy — numerical operations
Scikit-learn — preprocessing, encoding, scaling, and machine learning

The experiment also specifies Matplotlib and Seaborn among the required libraries for the practical work. """