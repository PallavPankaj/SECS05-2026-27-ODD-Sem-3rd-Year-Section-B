
#NAME - PALLAV PANKAJ
#ROLL- 40

# ============================================================
# Statistical Analysis and Hypothesis Testing using Python
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
#NAME - PALLAV PANKAJ
#ROLL- 40
# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

# Change this to your dataset filename
FILE_NAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(FILE_NAME)

print("=" * 60)
print("STATISTICAL ANALYSIS AND HYPOTHESIS TESTING")
print("=" * 60)

print("\nFirst 5 records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

#NAME - PALLAV PANKAJ
#ROLL- 40
# ------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("2. DESCRIPTIVE STATISTICS")
print("=" * 60)

# Select numerical columns
numeric_columns = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "JobSatisfaction",
    "EnvironmentSatisfaction"
]

# Mean
print("\nMEAN:")
print(df[numeric_columns].mean())

# Median
print("\nMEDIAN:")
print(df[numeric_columns].median())

# Mode
print("\nMODE:")
print(df[numeric_columns].mode().iloc[0])

# Variance
print("\nVARIANCE:")
print(df[numeric_columns].var())

# Standard Deviation
print("\nSTANDARD DEVIATION:")
print(df[numeric_columns].std())

# Complete descriptive statistics
print("\nCOMPLETE DESCRIPTIVE STATISTICS:")
print(df[numeric_columns].describe())
#NAME - PALLAV PANKAJ
#ROLL- 40

# ------------------------------------------------------------
# 3. PEARSON CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("3. PEARSON CORRELATION ANALYSIS")
print("=" * 60)

correlation_matrix = df[numeric_columns].corr(method="pearson")

print("\nPearson Correlation Matrix:")
print(correlation_matrix)

# Heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Pearson Correlation Matrix")
plt.tight_layout()
plt.show()
#NAME - PALLAV PANKAJ
#ROLL- 40

# ------------------------------------------------------------
# 4. INDEPENDENT SAMPLE T-TEST
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("4. INDEPENDENT SAMPLE T-TEST")
print("=" * 60)

# Business Problem:
# Is there a significant difference in Monthly Income
# between employees who leave the company and those who stay?

# H0:
# There is NO significant difference in monthly income
# between employees who leave and stay.

# H1:
# There IS a significant difference in monthly income
# between employees who leave and stay.

# Convert Attrition into two groups
left = df[df["Attrition"] == "Yes"]["MonthlyIncome"]
stayed = df[df["Attrition"] == "No"]["MonthlyIncome"]

print("\nNumber of employees who left:", len(left))
print("Number of employees who stayed:", len(stayed))

print("\nMean Monthly Income - Left:")
print(left.mean())

print("\nMean Monthly Income - Stayed:")
print(stayed.mean())
#NAME - PALLAV PANKAJ
#ROLL- 40

# Independent t-test
t_statistic, p_value = stats.ttest_ind(
    left,
    stayed,
    equal_var=False
)

print("\nT-Statistic:", t_statistic)
print("P-Value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("\nDecision: Reject H0")
    print("There is a statistically significant difference in")
    print("Monthly Income between employees who left and stayed.")
else:
    print("\nDecision: Fail to Reject H0")
    print("There is no statistically significant difference in")
    print("Monthly Income between employees who left and stayed.")

#NAME - PALLAV PANKAJ
#ROLL- 40
# ------------------------------------------------------------
# 5. ONE-WAY ANOVA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("5. ONE-WAY ANOVA")
print("=" * 60)

# Business Problem:
# Is there a significant difference in Job Satisfaction
# across different Job Roles?

# H0:
# Mean Job Satisfaction is the same across all Job Roles.

# H1:
# At least one Job Role has a different mean
# Job Satisfaction.

job_roles = df["JobRole"].unique()

groups = []

print("\nJob Roles:")
for role in job_roles:
    group = df[df["JobRole"] == role]["JobSatisfaction"]
    groups.append(group)

    print(
        role,
        "-> Mean Job Satisfaction:",
        round(group.mean(), 2),
        "| Count:",
        len(group)
    )

# Perform one-way ANOVA
f_statistic, anova_p_value = stats.f_oneway(*groups)

print("\nF-Statistic:", f_statistic)
print("P-Value:", anova_p_value)

if anova_p_value < alpha:
    print("\nDecision: Reject H0")
    print("There is a statistically significant difference")
    print("in Job Satisfaction among Job Roles.")
else:
    print("\nDecision: Fail to Reject H0")
    print("There is no statistically significant difference")
    print("in Job Satisfaction among Job Roles.")

#NAME - PALLAV PANKAJ
#ROLL- 40

# ------------------------------------------------------------
# 6. LINEAR REGRESSION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("6. LINEAR REGRESSION")
print("=" * 60)

# Business Problem:
# Analyze the relationship between YearsAtCompany
# and MonthlyIncome.

# Independent Variable (X)
X = df["YearsAtCompany"]

# Dependent Variable (Y)
Y = df["MonthlyIncome"]

# Add constant/intercept
X_with_constant = sm.add_constant(X)

# Build regression model
model = sm.OLS(Y, X_with_constant).fit()

print("\nREGRESSION SUMMARY:")
print(model.summary())

# Regression coefficients
print("\nRegression Coefficients:")
print(model.params)

# R-squared
print("\nR-Squared:")
print(model.rsquared)

# P-value
print("\nRegression P-Value:")
print(model.f_pvalue)

# Confidence intervals
print("\n95% Confidence Intervals:")
print(model.conf_int())

#NAME - PALLAV PANKAJ
#ROLL- 40

# ------------------------------------------------------------
# 7. REGRESSION VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.regplot(
    x="YearsAtCompany",
    y="MonthlyIncome",
    data=df,
    scatter_kws={"alpha": 0.5},
    line_kws={"linewidth": 2}
)

plt.title("Years at Company vs Monthly Income")
plt.xlabel("Years at Company")
plt.ylabel("Monthly Income")

plt.tight_layout()
plt.show()

#NAME - PALLAV PANKAJ
#ROLL- 40

# ------------------------------------------------------------
# 8. FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print(" FINAL INTERPRETATION")
print("=" * 60)

print("\n1. Descriptive statistics summarize the central tendency")
print("   and variation of the numerical variables.")

print("\n2. Pearson correlation measures the strength and direction")
print("   of the linear relationship between numerical variables.")

print("\n3. The independent-sample t-test compares the mean")
print("   Monthly Income of employees who left and stayed.")

print("\n4. One-way ANOVA determines whether Job Satisfaction")
print("   differs significantly across Job Roles.")

print("\n5. Linear regression analyzes the relationship between")
print("   Years at Company and Monthly Income.")

print("\n6. R-squared indicates how much variation in Monthly Income")
print("   is explained by Years at Company.")

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 60)

#NAME - PALLAV PANKAJ
#ROLL- 40

# QUESTION ANSWERS:-


# 1. What is the difference between descriptive statistics and inferential statistics?

# Answer:
# Descriptive statistics are used to summarize and describe data using measures such as mean, median, mode, variance, and standard deviation.
# Inferential statistics are used to draw conclusions or make predictions about a larger population based on sample data. Examples include t-tests, ANOVA, and regression analysis.


# 2. Explain the concepts of the Null Hypothesis (H₀) and Alternative Hypothesis (H₁).

# Answer:
# The Null Hypothesis (H₀) states that there is no significant difference, relationship, or effect in the data.
# The Alternative Hypothesis (H₁) states that there is a significant difference, relationship, or effect.

# For example:

# H₀: There is no significant difference in monthly income between employees who leave and stay.
# H₁: There is a significant difference in monthly income between employees who leave and stay.



# 3. What is a p-value? How is it used to make statistical decisions?

# Answer:
# A p-value indicates how likely it is to obtain the observed result if the null hypothesis is true.

# Usually, we use a significance level of 0.05:
# If p-value < 0.05 → Reject H₀.
# If p-value ≥ 0.05 → Fail to reject H₀.

# A smaller p-value provides stronger evidence against the null hypothesis.

#NAME - PALLAV PANKAJ
#ROLL- 40

# 4. Differentiate between a t-test and ANOVA. In which situations is each test applied?

# Answer:

# t-test	                                                  ANOVA
# Compares the means of two groups	                          Compares means of three or more groups
# Produces a t-statistic	                                  Produces an F-statistic
# Example: Income of employees who left vs stayed	          Example: Satisfaction across different job roles

# An independent-sample t-test is used when comparing two independent groups, while One-Way ANOVA is used to compare multiple groups.



# 5. What does the Pearson correlation coefficient indicate? What are its possible values?

# Answer:
# The Pearson correlation coefficient measures the strength and direction of the linear relationship between two numerical variables.

# Its value ranges from -1 to +1:

# +1 → Perfect positive correlation
# 0 → No linear correlation
# -1 → Perfect negative correlation

# A positive value means that as one variable increases, the other tends to increase. A negative value means that as one increases, the other tends to decrease.



# 6. Explain the significance of R² (Coefficient of Determination) in Linear Regression.

# Answer:
# R² (R-squared) measures the proportion of variation in the dependent variable that is explained by the independent variable(s) in a regression model.

# For example, if:

# R² = 0.70

# then approximately 70% of the variation in the dependent variable is explained by the regression model.

# A higher R² generally indicates a better fit, although a high R² alone does not prove that the model is appropriate.



# 7. Why is statistical analysis important before applying machine learning algorithms?

# Answer:
# Statistical analysis helps us:

# Understand the dataset.
# Identify relationships between variables.
# Detect patterns and unusual values.
# Check assumptions about the data.
# Identify important variables.
# Test whether observed relationships are statistically significant.

# Therefore, statistical analysis can help in making better decisions about data preprocessing, feature selection, and model building.



# 8. What assumptions should be satisfied before performing a t-test or ANOVA?

# Answer:
# Important assumptions include:

# Independence – observations should be independent.
# Normality – the data within groups should be approximately normally distributed.
# Homogeneity of variance – groups should have approximately equal variances, especially for the standard versions of these tests.
# The dependent variable should generally be numerical/continuous.

# If assumptions are seriously violated, appropriate alternatives or corrections may be required.




# 9. How can regression analysis help organizations in forecasting and decision-making?

# Answer:
# Regression analysis helps organizations understand relationships between variables and predict future outcomes.
# For example, a company can analyze the relationship between Years at Company and Monthly Income to understand how experience within the company relates to income.

# Regression can help organizations with:

# Sales forecasting
# Revenue prediction
# Employee analysis
# Cost estimation
# Demand forecasting
# Business planning

# The experiment specifically includes simple linear regression to analyze relationships such as Years of Experience and Monthly Income.



# 10. Give two real-world applications where hypothesis testing is commonly used in data analytics.

# Answer:

# 1. Employee analysis:
# A company can test whether there is a significant difference in monthly income between employees who leave and employees who stay.

# 2. Marketing analysis:
# A company can test whether a new advertising campaign produces a significant increase in sales compared with the previous campaign.

# Hypothesis testing helps organizations determine whether an observed difference is statistically significant rather than simply due to random variation.