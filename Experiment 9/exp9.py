
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#NAME - PALLAV PANKAJ
#ROLL NO -40
sns.set_theme(style="whitegrid")

FILE_NAME = "Superstore.csv"

if not os.path.exists(FILE_NAME):
    print("Dataset not found:", FILE_NAME)
    exit()

try:
    df = pd.read_csv(FILE_NAME, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(FILE_NAME, encoding="latin1")

df.columns = df.columns.str.strip()
df = df.drop_duplicates()

print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())

# Convert columns
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# KPIs
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order ID"].nunique()
profit_margin = (total_profit / total_sales) * 100

print("\n========== KPIs ==========")
print("Total Sales:", round(total_sales, 2))
print("Total Profit:", round(total_profit, 2))
print("Total Orders:", total_orders)
print("Profit Margin:", round(profit_margin, 2), "%")


#NAME - PALLAV PANKAJ
#ROLL NO -40
# 1. Sales by Category
category_sales = df.groupby("Category")["Sales"].sum()

print("\nSales by Category:\n", category_sales)

category_sales.plot(kind="bar", figsize=(8, 5))
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()


# 2. Profit by Category
category_profit = df.groupby("Category")["Profit"].sum()

print("\nProfit by Category:\n", category_profit)

category_profit.plot(kind="bar", figsize=(8, 5))
plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()


# 3. Monthly Sales Trend
monthly_sales = df.groupby(
    df["Order Date"].dt.to_period("M")
)["Sales"].sum()

plt.figure(figsize=(12, 5))
plt.plot(
    monthly_sales.index.astype(str),
    monthly_sales.values,
    marker="o"
)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


#NAME - PALLAV PANKAJ
#ROLL NO -40
# 4. Sales by Region
region_sales = df.groupby("Region")["Sales"].sum()

print("\nSales by Region:\n", region_sales)

region_sales.plot(kind="bar", figsize=(8, 5))
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 5. Profit by Region
region_profit = df.groupby("Region")["Profit"].sum()

print("\nProfit by Region:\n", region_profit)

region_profit.plot(kind="bar", figsize=(8, 5))
plt.title("Profit by Region")
plt.xlabel("Region")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 6. Region vs Category Heatmap
pivot = df.pivot_table(
    values="Profit",
    index="Region",
    columns="Category",
    aggfunc="sum"
)

plt.figure(figsize=(9, 6))
sns.heatmap(pivot, annot=True, fmt=".0f")
plt.title("Profit by Region and Category")
plt.tight_layout()
plt.show()


# 7. Top 10 Products
top_products = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Products:\n", top_products)

top_products.sort_values().plot(
    kind="barh",
    figsize=(10, 6)
)
plt.title("Top 10 Products by Sales")
plt.xlabel("Sales")
plt.ylabel("Product")
plt.tight_layout()
plt.show()


# 8. Loss-Making Products
product_profit = df.groupby("Product Name")["Profit"].sum()
loss_products = product_profit[product_profit < 0].sort_values().head(10)

print("\nLoss-Making Products:\n", loss_products)

loss_products.sort_values().plot(
    kind="barh",
    figsize=(10, 6)
)
plt.title("Top Loss-Making Products")
plt.xlabel("Profit")
plt.ylabel("Product")
plt.tight_layout()
plt.show()


#NAME - PALLAV PANKAJ
#ROLL NO -40
# 9. Sales vs Profit
plt.figure(figsize=(8, 6))
plt.scatter(df["Sales"], df["Profit"], alpha=0.5)
plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()


# 10. Correlation Heatmap
numeric = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))
sns.heatmap(numeric.corr(), annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Business Insights
best_category = category_sales.idxmax()
best_profit_category = category_profit.idxmax()
best_region = region_sales.idxmax()
best_profit_region = region_profit.idxmax()

print("\n========== BUSINESS INSIGHTS ==========")
print("Highest Sales Category:", best_category)
print("Highest Profit Category:", best_profit_category)
print("Highest Sales Region:", best_region)
print("Highest Profit Region:", best_profit_region)
print("Number of Loss-Making Products:", len(product_profit[product_profit < 0]))


print("\n========== RECOMMENDATIONS ==========")
print("1. Focus marketing on high-performing categories.")
print("2. Investigate loss-making products.")
print("3. Review pricing and discount strategies.")
print("4. Maintain inventory for high-sales products.")
print("5. Analyze weaker regions for improvement.")
print("6. Use sales trends for business and inventory planning.")

print("\n========== EXPERIMENT COMPLETED ==========")

#NAME - PALLAV PANKAJ
#ROLL NO -40


"""
QUESTION & ANSWERS:

1. What is data storytelling? How is it different from data visualization?

Answer:
Data storytelling is the process of using data, visualizations, and a narrative to explain important findings clearly. It helps the audience understand not only what the data shows but also why it is important.

Data visualization mainly represents data using charts, graphs, and plots, while data storytelling combines visualizations with explanations and context to communicate a complete message.


2. Why is storytelling important in business analytics and decision-making?

Answer:
Storytelling makes complex analytical results easier to understand. It helps business stakeholders identify important trends, problems, opportunities, and performance gaps. 
It also helps analysts communicate their findings clearly so that decisions can be based on data.


3. What are the essential components of an effective data story?

Answer:
The main components of an effective data story are:

Data – Reliable and relevant information.
Visualization – Charts or graphs that clearly represent the data.
Narrative – An explanation that connects the findings.
Context – Background information that helps explain the results.
Insights – Important findings obtained from the analysis.
Recommendations – Actions that can be taken based on the findings.


4. How do Key Performance Indicators (KPIs) enhance business reporting?

Answer:
KPIs provide measurable values that show how well a business or process is performing. They make reports more focused and easier to understand. Examples include total sales, total profit, number of orders, revenue growth, and profit margin. KPIs help stakeholders quickly monitor performance and identify areas that need attention.


5. Why should visualizations be arranged in a logical sequence while presenting analytical findings?

Answer:
Visualizations should be arranged logically so that the audience can easily follow the analysis. A good sequence can start with overall KPIs, followed by trends, category or regional analysis, detailed findings, and finally recommendations. This creates a clear flow from the problem to the findings and possible actions.


6. What factors should be considered while selecting visualizations for a business presentation?

Answer:
The following factors should be considered:

Type of data being presented
Purpose of the visualization
Audience and their level of understanding
Number of variables
Need to compare values or identify trends
Clarity and simplicity
Avoiding unnecessary visual elements

For example, bar charts are useful for comparisons, while line charts are suitable for showing trends over time.


7. Explain how dashboards and storytelling complement each other in Business Intelligence.

Answer:
Dashboards provide an interactive view of important business data through KPIs, charts, and graphs. Storytelling adds context by explaining what the visualizations mean and why the findings are important.

Therefore, dashboards help users explore the data, while storytelling helps them understand the main message and make informed decisions.


8. What challenges may arise while communicating analytical insights to non-technical stakeholders?

Answer:
Some common challenges are:

Complex technical terminology
Large amounts of information
Difficulty understanding statistical concepts
Lack of familiarity with analytical tools
Misinterpretation of charts
Difficulty connecting analytical findings with business objectives

These challenges can be reduced by using simple language, clear visualizations, relevant examples, and focusing on actionable insights.


9. Give two real-world examples where data storytelling has influenced business or policy decisions.

Answer:

Example 1 – Business:
A retail company can analyze sales data and create a data story showing that certain products have high sales but low profit. Management can then review pricing, discounts, and costs for those products.

Example 2 – Public Policy:
Government agencies can use data about disease cases, population, and geographical distribution to create visual reports. These insights can help policymakers identify areas requiring additional healthcare resources and plan appropriate actions.


10. How can effective data storytelling improve strategic planning and organizational performance?

Answer:
Effective data storytelling helps organizations understand their current performance, identify trends and problems, and recognize opportunities. It allows managers to connect analytical findings with business objectives and take appropriate actions.

It can improve strategic planning by supporting better resource allocation, performance monitoring, risk identification, and decision-making. This ultimately helps organizations make more informed, data-driven plans."""