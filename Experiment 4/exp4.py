#NAME - PALLAV PANKAJ
#ROLL-40

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# EXPERIMENT NO. 4
# ADVANCED DATA VISUALIZATION USING MATPLOTLIB AND SEABORN
# ============================================================

# Load dataset
FILE_NAME = "Superstore.csv"

df = pd.read_csv(FILE_NAME, encoding="latin1")
#NAME - PALLAV PANKAJ
#ROLL-40

print("=" * 60)
print("ADVANCED DATA VISUALIZATION")
print("=" * 60)

print("\nFirst 5 records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# Data preprocessing
# ------------------------------------------------------------

#NAME - PALLAV PANKAJ
#ROLL-40

# Convert Order Date into datetime format
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)

# Remove rows with invalid dates
df = df.dropna(subset=["Order Date"])

# Create output folder
os.makedirs("results", exist_ok=True)

sns.set_theme(style="whitegrid")


# 1. BAR CHART - SALES BY REGION

#NAME - PALLAV PANKAJ
#ROLL-40

region_sales = df.groupby("Region")["Sales"].sum().sort_values(
    ascending=False
)

plt.figure(figsize=(9, 6))

sns.barplot(
    x=region_sales.index,
    y=region_sales.values
)

plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("results/01_sales_by_region.png")
plt.show()


# ============================================================
# 2. BAR CHART - SALES BY CATEGORY
# ============================================================

category_sales = df.groupby("Category")["Sales"].sum().sort_values(
    ascending=False
)

plt.figure(figsize=(8, 6))

sns.barplot(
    x=category_sales.index,
    y=category_sales.values
)

plt.title("Total Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")

plt.tight_layout()

plt.savefig("results/02_sales_by_category.png")
plt.show()


# ============================================================
# 3. LINE CHART - MONTHLY SALES TREND
# ============================================================
#NAME - PALLAV PANKAJ
#ROLL-40
monthly_sales = (
    df.set_index("Order Date")
      .resample("ME")["Sales"]
      .sum()
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()

plt.savefig("results/03_monthly_sales.png")
plt.show()


# ============================================================
# 4. HISTOGRAM - SALES DISTRIBUTION
# ============================================================

#NAME - PALLAV PANKAJ
#ROLL-40

plt.figure(figsize=(9, 6))

sns.histplot(
    df["Sales"],
    bins=30,
    kde=True
)

plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("results/04_sales_histogram.png")
plt.show()


# ============================================================
# 5. BOX PLOT - SALES BY CATEGORY
# ============================================================

plt.figure(figsize=(9, 6))

sns.boxplot(
    x="Category",
    y="Sales",
    data=df
)

plt.title("Sales Distribution by Category")
plt.xlabel("Product Category")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig("results/05_sales_boxplot.png")
plt.show()

#NAME - PALLAV PANKAJ
#ROLL-40

# ============================================================
# 6. SCATTER PLOT - SALES VS PROFIT
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x="Sales",
    y="Profit",
    data=df,
    alpha=0.6
)

plt.title("Relationship Between Sales and Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")

plt.tight_layout()

plt.savefig("results/06_sales_profit_scatter.png")
plt.show()


# ============================================================
# 7. SCATTER PLOT - DISCOUNT VS PROFIT
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x="Discount",
    y="Profit",
    data=df,
    alpha=0.6
)

plt.title("Relationship Between Discount and Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")

plt.tight_layout()

plt.savefig("results/07_discount_profit_scatter.png")
plt.show()

#NAME - PALLAV PANKAJ
#ROLL-40

# ============================================================
# 8. CORRELATION HEATMAP
# ============================================================

numeric_columns = [
    "Sales",
    "Quantity",
    "Discount",
    "Profit"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("results/08_correlation_heatmap.png")
plt.show()


# ============================================================
# 9. SALES BY REGION AND CATEGORY
# ============================================================

region_category = pd.pivot_table(
    df,
    values="Sales",
    index="Region",
    columns="Category",
    aggfunc="sum"
)

plt.figure(figsize=(10, 6))

region_category.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Sales by Region and Category")
plt.xlabel("Region")
plt.ylabel("Sales")

plt.xticks(rotation=0)
plt.legend(title="Category")

plt.tight_layout()

plt.savefig("results/09_region_category_sales.png")
plt.show()

#NAME - PALLAV PANKAJ
#ROLL-40

# ============================================================
# 10. PROFIT BY CATEGORY
# ============================================================

category_profit = df.groupby("Category")["Profit"].sum()

plt.figure(figsize=(8, 6))

sns.barplot(
    x=category_profit.index,
    y=category_profit.values
)

plt.title("Total Profit by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Profit")

plt.tight_layout()

plt.savefig("results/10_profit_by_category.png")
plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nTotal Sales:")
print(df["Sales"].sum())

print("\nTotal Profit:")
print(df["Profit"].sum())

print("\nBest Performing Region:")
print(region_sales.idxmax())

print("\nBest Selling Category:")
print(category_sales.idxmax())

print("\nMost Profitable Category:")
print(category_profit.idxmax())

print("\nAll charts have been saved in the 'results' folder.")

#NAME - PALLAV PANKAJ
#ROLL-40
