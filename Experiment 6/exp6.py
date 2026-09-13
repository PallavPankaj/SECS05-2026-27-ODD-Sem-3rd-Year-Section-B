# EXPERIMENT NO. 6
# Predictive Analytics using Linear Regression
# Model Performance Evaluation

#NAME - PALLAV PANKAJ
#ROLL - 40

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

housing = fetch_california_housing(as_frame=True)

df = housing.frame

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())
#NAME - PALLAV PANKAJ
#ROLL - 40

# ---------------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# ---------------------------------------------------------

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("\nFeatures:")
print(X.columns)

print("\nTarget: MedHouseVal")
#NAME - PALLAV PANKAJ
#ROLL - 40

# ---------------------------------------------------------
# 3. SPLIT DATASET INTO TRAINING AND TESTING SETS
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------------------------------------------------------
# 4. CREATE LINEAR REGRESSION MODEL
# ---------------------------------------------------------

model = LinearRegression()

# Train model
model.fit(X_train, y_train)
#NAME - PALLAV PANKAJ
#ROLL - 40

# ---------------------------------------------------------
# 5. MAKE PREDICTIONS
# ---------------------------------------------------------

y_pred = model.predict(X_test)

print("\nFirst 10 Actual Values:")
print(y_test.head(10).values)

print("\nFirst 10 Predicted Values:")
print(y_pred[:10])

# ---------------------------------------------------------
# 6. MODEL PERFORMANCE EVALUATION
# ---------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)
#NAME - PALLAV PANKAJ
#ROLL - 40

# ---------------------------------------------------------
# 7. REGRESSION COEFFICIENTS
# ---------------------------------------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\n========== REGRESSION COEFFICIENTS ==========")
print(coefficients)

print("\nIntercept:", model.intercept_)

# ---------------------------------------------------------
# 8. ACTUAL VS PREDICTED VALUES
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

# Perfect prediction line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.tight_layout()
plt.show()
#NAME - PALLAV PANKAJ
#ROLL - 40

# ---------------------------------------------------------
# 9. RESIDUAL PLOT
# ---------------------------------------------------------

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(y_pred, residuals, alpha=0.5)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10. CONCLUSION
# ---------------------------------------------------------

print("\n========== CONCLUSION ==========")

if r2 >= 0.70:
    print("The Linear Regression model provides a reasonably good fit.")
elif r2 >= 0.50:
    print("The Linear Regression model provides a moderate fit.")
else:
    print("The Linear Regression model has limited predictive performance.")

print("The model can be used to predict continuous house-price values.")

#NAME - PALLAV PANKAJ
#ROLL - 40


# Q&A
# 1. What is Predictive Analytics, and how is it used in real-world applications?

# Answer:
# Predictive Analytics is the process of using historical data, statistical techniques, and machine-learning algorithms to predict future outcomes. It helps organizations make better decisions based on patterns in previous data.

# Applications: sales forecasting, house-price prediction, demand forecasting, fraud detection, customer behavior prediction, and risk analysis.

# 2. Explain the working principle of the Linear Regression algorithm.

# Answer:
# Linear Regression establishes a mathematical relationship between independent variables and a continuous dependent variable.

# For simple linear regression:

# $$ Y = b_0 + b_1X $$

# where:

# Y = predicted value
# X = independent variable
# b₀ = intercept
# b₁ = regression coefficient

# For multiple linear regression:


# The algorithm finds coefficients that minimize the difference between actual and predicted values.

# 3. Differentiate between dependent and independent variables with suitable examples.
# Independent Variable	Dependent Variable
# Input/predictor variable	Output/target variable
# Used to make predictions	Value being predicted
# Usually represented by X	Usually represented by Y
# Example: house area	Example: house price

# For example, when predicting house prices, income, house age, number of rooms, latitude and longitude can be independent variables, while house price is the dependent variable.

# 4. Why is it necessary to split the dataset into training and testing sets?

# Answer:
# The dataset is divided into training and testing sets to determine whether the model can generalize to unseen data.

# Training data: Used to train the model.
# Testing data: Used to evaluate the trained model.

# An 80:20 split is commonly used and is specifically suggested in the experiment instructions.

# 5. What is the significance of the R² Score in regression analysis?

# Answer:
# R² Score measures how much of the variation in the dependent variable is explained by the regression model.

# $$ R^2 = 1-\frac{SS_{res}}{SS_{tot}} $$

# Generally:

# R² = 1: Perfect prediction
# R² close to 1: Very good fit
# R² close to 0: Weak explanatory power
# Negative R²: Model performs worse than a simple mean-based prediction

# A higher R² generally indicates a better model fit.

# 6. Differentiate between MAE, MSE, and RMSE. Which metric is more sensitive to large prediction errors?
# Metric	Meaning	Characteristics
# MAE	Mean Absolute Error	Average absolute prediction error
# MSE	Mean Squared Error	Average squared prediction error
# RMSE	Root Mean Squared Error	Square root of MSE

# Formulas:

# $$ MAE=\frac{1}{n}\sum |y-\hat y| $$ $$ MSE=\frac{1}{n}\sum(y-\hat y)^2 $$ $$ RMSE=\sqrt{MSE} $$

# MSE and RMSE are more sensitive to large prediction errors because the errors are squared. The experiment specifically requires evaluation using MAE, MSE, RMSE and R².

# 7. What assumptions should be satisfied before applying Linear Regression?

# Answer:

# The major assumptions are:

# Linearity — relationship between predictors and target should be approximately linear.
# Independence — observations should be independent.
# Homoscedasticity — residual variance should remain approximately constant.
# Normality of residuals — residuals should be approximately normally distributed for reliable statistical inference.
# Low multicollinearity — independent variables should not be highly correlated with each other.
# 8. How can overfitting and underfitting affect the performance of a regression model?

# Answer:

# Overfitting:
# The model learns the training data too closely, including noise. It performs well on training data but poorly on unseen testing data.

# Underfitting:
# The model is too simple to capture important relationships in the data. It performs poorly on both training and testing data.

# A good model should balance complexity and generalization.

# 9. Mention any three real-world applications of Linear Regression in business or industry.

# Answer:

# House-price prediction — predicting property prices based on features such as location, size and number of rooms.
# Sales forecasting — predicting future sales from historical sales and business factors.
# Demand forecasting — estimating future product demand to support inventory planning.

# The lab sheet also specifically identifies house-price prediction and advertising sales as suitable regression problems.

# 10. How can feature selection improve the accuracy and interpretability of a predictive model?

# Answer:
# Feature selection removes irrelevant or redundant variables from the model.

# It can:

# Reduce noise.
# Reduce overfitting.
# Improve model performance.
# Reduce computational complexity.
# Make the model easier to understand.
# Make regression coefficients easier to interpret.

# The experiment specifically asks students to analyze regression coefficients to understand the impact of independent variables on the target.