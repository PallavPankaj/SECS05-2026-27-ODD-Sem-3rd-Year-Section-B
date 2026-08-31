import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Name: Pallav Pankaj
# Roll No.: 40

df = pd.read_csv("Superstore.csv", encoding="latin1")
print("Dataset loaded successfully!")
print("Shape of dataset:", df.shape)

# Name: Pallav Pankaj
# Roll No.: 40

print("Column Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nSummary Statistics:")
print(df.describe())


print("Missing Values:")
print(df.isnull().sum())

# Name: Pallav Pankaj
# Roll No.: 40

numerical_columns = df.select_dtypes(include=np.number).columns

df[numerical_columns].hist(
    figsize=(12, 8),
    bins=20
)

plt.tight_layout()
plt.show()

# Name: Pallav Pankaj
# Roll No.: 40

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Category"
)

plt.title("Distribution of Product Categories")
plt.xlabel("Category")
plt.ylabel("Count")

plt.show()

# Name: Pallav Pankaj
# Roll No.: 40

numeric_df = df.select_dtypes(include=np.number)

correlation = numeric_df.corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

# Name: Pallav Pankaj
# Roll No.: 40


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    y="Sales"
)

plt.title("Box Plot of Sales")
plt.ylabel("Sales")

plt.show()

# Name: Pallav Pankaj
# Roll No.: 40

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Sales",
    y="Profit"
)

plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")

plt.show()

# Name: Pallav Pankaj
# Roll No.: 40

print("Sales Statistics:")
print(df["Sales"].describe())

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Sales",
    kde=True
)

plt.title("Sales Distribution")

plt.show()

# Name: Pallav Pankaj
# Roll No.: 40

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Sales",
    y="Profit",
    hue="Category"
)

plt.title("Sales vs Profit by Category")

plt.show()

# Name: Pallav Pankaj
# Roll No.: 40

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Sales",
    y="Profit",
    hue="Category",
    size="Quantity"
)

plt.title("Sales, Profit, Category and Quantity")

plt.show()

# Name: Pallav Pankaj
# Roll No.: 40

print("Average Sales:", df["Sales"].mean())
print("Average Profit:", df["Profit"].mean())
print("Maximum Sales:", df["Sales"].max())
print("Maximum Profit:", df["Profit"].max())

print("\nMost Common Category:")
print(df["Category"].mode()[0])

print("\nMost Common Region:")
print(df["Region"].mode()[0])

# Name: Pallav Pankaj
# Roll No.: 40


# """  Questions and Answers

# 1. What is Exploratory Data Analysis (EDA), and why is it performed before machine learning?

# Answer:
# Exploratory Data Analysis (EDA) is the process of examining and analyzing a dataset to understand its structure, distributions, relationships, patterns, and anomalies.
# EDA is performed before machine learning because it helps identify missing values, outliers, skewness, relationships between variables, and other data-quality issues. It allows us to understand the data before building a predictive model. The lab sheet describes EDA as a critical stage before predictive modeling and business decisions.


# 2. Differentiate between univariate, bivariate, and multivariate analysis with suitable examples.

# Analysis	    Meaning	                                  Example
# Univariate	    Analysis of one variable	              Distribution of Sales
# Bivariate	    Analysis of two variables	              Sales vs Profit
# Multivariate	Analysis of more than two variables	      Sales, Profit, Category and Quantity

# Univariate analysis studies one variable.
# Bivariate analysis studies the relationship between two variables.
# Multivariate analysis studies multiple variables simultaneously.


# 3. What insights can be obtained from a correlation heatmap?

# A correlation heatmap shows the strength and direction of relationships between numerical variables.

# It can help identify:
# Positive relationships
# Negative relationships
# Weak relationships
# Strongly correlated features
# Potentially redundant features

# The experiment specifically asks students to use a correlation matrix and heatmap to identify relationships between numerical variables.


# 4. Explain the purpose of histograms, box plots, and scatter plots in EDA.

# Histogram:
# Used to understand the distribution, frequency, central tendency, and spread of numerical data.

# Box Plot:
# Used to visualize the median, spread, quartiles, and potential outliers.

# Scatter Plot:
# Used to study the relationship between two numerical variables.

# These three visualizations are explicitly included in the experiment's required analysis.


# 5. How can EDA help identify data quality issues before analysis?

# EDA can identify:

# Missing values
# Duplicate records
# Outliers
# Incorrect data types
# Unusual values
# Skewed distributions
# Inconsistent categories

# By identifying these problems before analysis, the dataset can be cleaned and prepared for further processing.


# 6. Why is correlation important in predictive analytics? Can correlation imply causation?

# Correlation is important because it indicates how two numerical variables are related. Highly correlated features can sometimes be useful for prediction or may indicate redundant features.
# However, correlation does not imply causation.
# For example, two variables may increase together because of another hidden factor. Therefore, a correlation alone cannot prove that one variable causes another.


# 7. Which visualization would you use to analyze categorical and numerical variables? Justify your choice.

# For categorical variables, I would use:
# Bar charts
# Count plots

# For numerical variables, I would use:
# Histograms
# Box plots
# Scatter plots

# For example, a count plot can show the number of products in each category, while a scatter plot can show the relationship between Sales and Profit.
# The experiment specifically recommends bar charts/count plots for categorical variables and histograms, box plots, and scatter plots for numerical analysis.


# 8. What business insights can be derived from the Netflix (or Superstore/HR Analytics) dataset through EDA?

# For a Superstore dataset, EDA can provide insights such as:

# Which product categories have the highest sales
# Which regions generate the most sales
# Which products generate higher profits
# The relationship between sales and profit
# Potentially unusual or extreme sales values
# Distribution of orders across different categories and regions

# These insights can support data-driven business decisions, which is one of the purposes stated in the experiment.


# 9. How does EDA contribute to feature selection and model building?

# EDA helps understand which variables are useful for prediction.
# Correlation analysis can identify highly related features. Distribution analysis can identify skewed variables, while outlier analysis can identify problematic observations.

# Therefore, EDA helps in:

# Selecting useful features
# Detecting redundant features
# Identifying outliers
# Understanding relationships
# Preparing data for machine learning


# 10. What challenges might arise while performing EDA on large-scale real-world datasets?

# Some common challenges include:

# Large dataset size
# High computational requirements
# Missing or inconsistent data
# Large numbers of features
# Difficulty in visualizing huge datasets
# Outliers and noisy data
# Complex relationships between variables
# Long processing times

# Therefore, large datasets may require efficient data-processing techniques and appropriate visualization methods.



# Conclusion:

# Exploratory Data Analysis was successfully performed on a real-world business dataset using Python.
# The dataset was explored using descriptive statistics, histograms, count plots, correlation heatmaps, box plots, and scatter plots. Univariate, bivariate, and multivariate analyses were also performed. 
# EDA helped identify patterns, relationships, distributions, and potential anomalies and provided useful insights for further analysis and business decision-making. 
# This matches the experiment's required objective of understanding the dataset and generating meaningful business insights."""