# EXPERIMENT 10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#NAME - PALLAV PANKAJ
#ROLL NO -40
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from scipy.stats import pearsonr

#NAME - PALLAV PANKAJ
#ROLL NO -40
# ---------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("Superstore.csv", encoding="latin1")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------
# 2. DATASET INFORMATION
# ---------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# ---------------------------------------------------
# 3. CHECK MISSING VALUES
# ---------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# ---------------------------------------------------
# 4. REMOVE DUPLICATES
# ---------------------------------------------------

print("\nDuplicate Records:", df.duplicated().sum())

df = df.drop_duplicates()

# ---------------------------------------------------
# 5. HANDLE MISSING VALUES
# ---------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

categorical_columns = df.select_dtypes(
    include="object"
).columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values handled.")

#NAME - PALLAV PANKAJ
#ROLL NO -40
# ---------------------------------------------------
# 6. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Sales"],
    kde=True
)

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 7. CATEGORY ANALYSIS
# ---------------------------------------------------

if "Category" in df.columns:

    category_sales = df.groupby(
        "Category"
    )["Sales"].sum()

    plt.figure(figsize=(8, 5))

    category_sales.plot(
        kind="bar"
    )

    plt.title("Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Sales")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------
# 8. REGION ANALYSIS
# ---------------------------------------------------

if "Region" in df.columns:

    region_sales = df.groupby(
        "Region"
    )["Sales"].sum()

    plt.figure(figsize=(8, 5))

    region_sales.plot(
        kind="bar"
    )

    plt.title("Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Sales")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------
# 9. CORRELATION ANALYSIS
# ---------------------------------------------------

numeric_df = df.select_dtypes(
    include=np.number
)

print("\nCorrelation Matrix:")
print(numeric_df.corr())

plt.figure(figsize=(10, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 10. STATISTICAL ANALYSIS
# ---------------------------------------------------

if "Quantity" in df.columns:

    correlation, p_value = pearsonr(
        df["Quantity"],
        df["Sales"]
    )

    print("\n========== STATISTICAL ANALYSIS ==========")

    print(
        "Correlation between Quantity and Sales:",
        round(correlation, 4)
    )

    print(
        "P-value:",
        round(p_value, 6)
    )

    if p_value < 0.05:
        print(
            "The relationship is statistically significant."
        )
    else:
        print(
            "The relationship is not statistically significant."
        )

#NAME - PALLAV PANKAJ
#ROLL NO -40
# ---------------------------------------------------
# 11. MACHINE LEARNING PREPARATION
# ---------------------------------------------------

# Select useful columns

features = []

possible_features = [
    "Quantity",
    "Discount",
    "Profit"
]

for col in possible_features:
    if col in df.columns:
        features.append(col)

# Add categorical columns if available

categorical_features = []

possible_categorical = [
    "Category",
    "Sub-Category",
    "Region",
    "Segment"
]

for col in possible_categorical:
    if col in df.columns:
        categorical_features.append(col)

#NAME - PALLAV PANKAJ
#ROLL NO -40
# ---------------------------------------------------
# 12. ENCODE CATEGORICAL VARIABLES
# ---------------------------------------------------

model_df = df[features + categorical_features + ["Sales"]].copy()

encoder = LabelEncoder()

for col in categorical_features:

    model_df[col] = encoder.fit_transform(
        model_df[col].astype(str)
    )

# ---------------------------------------------------
# 13. DEFINE X AND Y
# ---------------------------------------------------

X = model_df.drop(
    "Sales",
    axis=1
)

y = model_df["Sales"]

print("\nFeatures used:")
print(X.columns.tolist())

# ---------------------------------------------------
# 14. TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# ---------------------------------------------------
# 15. RANDOM FOREST MODEL
# ---------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nRandom Forest model trained successfully.")

# ---------------------------------------------------
# 16. PREDICTION
# ---------------------------------------------------

y_pred = model.predict(
    X_test
)

# ---------------------------------------------------
# 17. MODEL EVALUATION
# ---------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n========== MODEL EVALUATION ==========")

print("Mean Absolute Error :", round(mae, 2))
print("Root Mean Squared Error :", round(rmse, 2))
print("R² Score :", round(r2, 4))

# ---------------------------------------------------
# 18. ACTUAL VS PREDICTED
# ---------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title(
    "Actual Sales vs Predicted Sales"
)

plt.tight_layout()
plt.show()

#NAME - PALLAV PANKAJ
#ROLL NO -40
# ---------------------------------------------------
# 19. FEATURE IMPORTANCE
# ---------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8, 5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance")

plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#NAME - PALLAV PANKAJ
#ROLL NO -40
# ---------------------------------------------------
# 20. BUSINESS INSIGHTS
# ---------------------------------------------------

print("\n========== BUSINESS INSIGHTS ==========")

print("1. Sales patterns can be analyzed using category, region, quantity and profit.")

print("2. Correlation analysis helps identify relationships between numerical variables.")

print("3. The machine learning model can estimate sales using selected business features.")

print("4. Feature importance identifies variables that contribute more to sales prediction.")

print("5. Model evaluation metrics indicate how well the predictive model performs.")

# ---------------------------------------------------
# 21. RECOMMENDATIONS
# ---------------------------------------------------

print("\n========== BUSINESS RECOMMENDATIONS ==========")
print("1. Use sales predictions to support inventory planning.")
print("2. Focus on important features identified by the model.")
print("3. Analyze low-performing categories and regions.")
print("4. Use historical sales patterns for business planning.")
print("5. Retrain the model regularly when new sales data becomes available.")
print("\nExperiment 10 completed successfully!")


#NAME - PALLAV PANKAJ
#ROLL NO -40

#QUESTIONS AND ANSWERS

# Q1. What are the major stages involved in an end-to-end data analytics project?
# Answer:
# The major stages are data collection, data preprocessing, exploratory data analysis, statistical analysis, visualization, machine learning, model evaluation, interpretation, and presentation of results.

# Q2. Why is data preprocessing considered the foundation of successful analytics and machine learning?
# Answer:
# Data preprocessing improves data quality by handling missing values, duplicates, outliers, and categorical variables. Clean and properly prepared data helps analytical methods and machine learning models produce more reliable results.

# Q3. How does Exploratory Data Analysis (EDA) contribute to model development?
# Answer:
# EDA helps understand the structure and characteristics of the dataset. It identifies trends, relationships, outliers, and important variables. This information helps in selecting useful features and choosing an appropriate machine learning model.

# Q4. What factors should be considered while selecting a machine learning algorithm for a business problem?
# Answer:
# The type of problem, size and structure of the dataset, target variable, available features, required accuracy, interpretability, computational resources, and business requirements should be considered. The algorithm should match the problem, such as regression for continuous values or classification for categories.

# Q5. Why is model evaluation essential before deploying a predictive model?
# Answer:
# Model evaluation determines how accurately and reliably a model performs on data. It helps identify errors and overfitting and ensures that the model is suitable for practical use. Different metrics such as Accuracy, Precision, Recall, F1-Score, MAE, RMSE, or R² can be used depending on the model.

# Q6. Explain how Business Intelligence tools complement machine learning in data analytics projects.
# Answer:
# Business Intelligence tools such as Tableau and Power BI help present data, KPIs, model results, and trends through interactive dashboards. Machine learning provides predictions or classifications, while BI tools make these results easier for business users to understand and use.

# Q7. What challenges are commonly encountered while working with real-world datasets?
# Answer:
# Common challenges include missing values, duplicate records, outliers, inconsistent data, categorical variables, incorrect formats, noisy data, and insufficient data. These issues must be addressed during preprocessing before analysis and modeling.

# Q8. How can analytical insights be converted into actionable business recommendations?
# Answer:
# Analysts should first identify important patterns and business problems from the results. These findings can then be connected to specific actions, such as improving marketing, managing inventory, reducing customer churn, or improving employee retention. Recommendations should be practical and supported by data.

# Q9. Why is effective presentation and data storytelling important in analytics projects?
# Answer:
# Effective presentation makes analytical findings easier to understand. Data storytelling connects visualizations with a clear explanation of the problem, findings, and recommendations. It helps decision-makers use analytical results effectively.

# Q10. Suggest future improvements or advanced techniques that could enhance the accuracy and effectiveness of the developed analytics solution.
# Answer:
# Future improvements can include feature engineering, hyperparameter tuning, ensemble models, cross-validation, advanced machine learning algorithms, automated data pipelines, real-time dashboards, and larger datasets. Model performance can also be improved by continuously monitoring and updating the model with new data.