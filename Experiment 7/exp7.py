# ============================================================
# EXPERIMENT 7
# Customer Segmentation using K-Means Clustering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

# Keep your CSV file in the same folder as this Python file.
# Change the filename if required.

FILE_NAME = "Mall_Customers.csv"

df = pd.read_csv(FILE_NAME)

print("\n" + "=" * 60)
print("CUSTOMER SEGMENTATION USING K-MEANS")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 2. DATA PREPROCESSING
# ------------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows containing missing values
df = df.dropna()

print("\nShape after preprocessing:")
print(df.shape)

# ------------------------------------------------------------
# 3. SELECT FEATURES
# ------------------------------------------------------------

# Standard Mall Customers dataset columns:
# Age, Annual Income (k$), Spending Score (1-100)

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

print("\nSelected features:")
print(X.head())

# ------------------------------------------------------------
# 4. FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled data:")
print(X_scaled[:5])

# ------------------------------------------------------------
# 5. ELBOW METHOD
# ------------------------------------------------------------

inertia = []

K_range = range(2, 11)

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    K_range,
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(K_range))
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. SILHOUETTE SCORE
# ------------------------------------------------------------

silhouette_scores = []

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    silhouette_scores.append(score)

print("\nSilhouette Scores:")

for k, score in zip(K_range, silhouette_scores):

    print(
        f"K = {k} --> "
        f"Silhouette Score = {score:.4f}"
    )

# Select K having highest silhouette score
best_k = list(K_range)[
    np.argmax(silhouette_scores)
]

print("\nBest K according to Silhouette Score:", best_k)

# ------------------------------------------------------------
# 7. APPLY K-MEANS
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# ------------------------------------------------------------
# 8. CLUSTER CENTROIDS
# ------------------------------------------------------------

centroids_scaled = kmeans.cluster_centers_

# Convert centroids back to original feature scale
centroids = scaler.inverse_transform(
    centroids_scaled
)

centroid_df = pd.DataFrame(
    centroids,
    columns=features
)

print("\nCluster Centroids:")
print(centroid_df)

# ------------------------------------------------------------
# 9. DISPLAY CLUSTER COUNTS
# ------------------------------------------------------------

print("\nNumber of customers in each cluster:")

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# 10. CLUSTER ANALYSIS
# ------------------------------------------------------------

print("\nCluster Characteristics:")
print("=" * 60)

cluster_summary = df.groupby("Cluster")[features].mean()

print(cluster_summary.round(2))

# ------------------------------------------------------------
# 11. VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[
        df["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        label=f"Cluster {cluster}",
        s=60
    )

# Plot centroids
plt.scatter(
    centroid_df["Annual Income (k$)"],
    centroid_df["Spending Score (1-100)"],
    marker="X",
    s=250,
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12. SAVE RESULT
# ------------------------------------------------------------

df.to_csv(
    "customer_segments.csv",
    index=False
)

print("\nClustered dataset saved as:")
print("customer_segments.csv")

print("\nExperiment completed successfully.")

# Q&A
# 1. What is clustering? How does it differ from classification?

# Clustering is an unsupervised machine-learning technique that groups similar data points into clusters.

# Classification is a supervised learning technique that assigns data points to predefined classes.

# Clustering	Classification
# Unsupervised learning	Supervised learning
# No predefined labels	Uses predefined labels
# Finds natural groups	Predicts known classes
# Example: customer segmentation	Example: spam detection
# 2. Explain the working principle of the K-Means Clustering algorithm.

# K-Means works as follows:

# Select the value of K.
# Initialize K centroids.
# Calculate the distance between each point and every centroid.
# Assign each point to its nearest centroid.
# Calculate new centroids.
# Repeat the process until the clusters become stable.

# The objective is to create clusters in which points are as close as possible to their respective centroids.

# 3. Why is feature scaling important before applying K-Means?

# K-Means uses distance calculations. If features have different scales, a feature with larger numerical values can dominate the distance calculation.

# For example, Annual Income may range around 15–140, while another feature may have a much smaller range.

# Standardization makes the features comparable and ensures that all selected variables contribute more fairly to clustering. The experiment explicitly requires normalization or standardization before clustering.

# 4. What is the Elbow Method?

# The Elbow Method is used to determine a suitable number of clusters.

# It calculates the within-cluster sum of squares/inertia for different values of K.

# The value of K where the decrease in inertia starts becoming significantly smaller is called the elbow point.

# The experiment requires using the Elbow Method to determine the optimal number of clusters.

# 5. What is the Silhouette Score?

# The Silhouette Score measures how well data points fit within their assigned clusters.

# Its value generally ranges from -1 to +1:

# Close to +1 → well-separated clusters
# Around 0 → overlapping clusters
# Negative → points may be assigned to the wrong cluster

# A higher score generally indicates better-defined clustering.

# The experiment specifies using the Silhouette Score to validate the selected K.

# 6. Why is K-Means an unsupervised algorithm?

# K-Means is unsupervised because the input dataset does not require predefined class labels.

# The algorithm itself discovers groups based on similarities between observations.

# For customer segmentation, we do not tell the algorithm which customer belongs to which segment beforehand. K-Means discovers the segments automatically.

# 7. Mention four real-world applications of customer segmentation.

# Four applications are:

# Targeted marketing campaigns
# Personalized product recommendations
# Customer retention
# Customer engagement improvement

# These applications align with the purpose of segmentation described in the experiment sheet.

# 8. What are the limitations of K-Means?

# Major limitations include:

# The number of clusters K must be selected beforehand.
# Results can be affected by the initial centroid positions.
# It can be sensitive to outliers.
# It works best when clusters have relatively suitable shapes and separations.
# Different feature scales can distort results if scaling is not performed.
# It may not perform well when clusters have very different densities or complex shapes.
# 9. How can businesses use customer segmentation to improve marketing and customer retention?

# Businesses can divide customers into groups based on characteristics such as income, age, spending behavior, or purchasing patterns.

# They can then:

# Send personalized offers.
# Give special discounts to valuable customers.
# Recommend relevant products.
# Create targeted advertising campaigns.
# Identify customers with low engagement.
# Develop retention strategies for important customer groups.

# The experiment specifically asks students to analyze customer groups and suggest suitable marketing strategies for each segment.

# 10. Compare K-Means and Hierarchical Clustering.

# K-Means	                            Hierarchical Clustering
# Partitional clustering	            Hierarchical clustering
# Requires K to be specified        	Does not necessarily require K initially
# Uses centroids	                    Uses a hierarchy/tree structure
# Generally faster for large datasets	Can be computationally expensive
# Produces separate clusters	        Produces a dendrogram
# Good for large datasets	            Useful for exploring relationships between groups
# Iteratively updates centroids	        Iteratively merges or splits clusters