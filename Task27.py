import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons, make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

st.set_page_config(page_title="Task27", layout="wide")

st.title("Task 27")
st.subheader("KMeans, DBSCAN and PCA using Streamlit")


# Q1


st.header("Q1. Creating Non-Linear Dataset")

X, y = make_moons(
    n_samples=500,
    noise=0.05,
    random_state=42
)

df = pd.DataFrame(
    X,
    columns=["Feature1", "Feature2"]
)

st.write("### First 10 Rows")
st.dataframe(df.head(10))

st.write("### Shape of Dataset")
st.write(df.shape)

# Q2

st.header("Q2. Standard Scaling")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=["Feature1", "Feature2"]
)

st.write("### First 5 Rows of Scaled Data")
st.dataframe(scaled_df.head())


# Q3


st.header("Q3. KMeans Clustering")

kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(
    df["Feature1"],
    df["Feature2"],
    c=df["kmeans_cluster"],
    cmap="viridis"
)

ax.set_title("KMeans Clustering")
ax.set_xlabel("Feature1")
ax.set_ylabel("Feature2")

st.pyplot(fig)


# Q4


st.header("Q4. DBSCAN Clustering")

dbscan = DBSCAN(
    eps=0.3,
    min_samples=5
)

df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)

fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(
    df["Feature1"],
    df["Feature2"],
    c=df["dbscan_cluster"],
    cmap="rainbow"
)

ax.set_title("DBSCAN Clustering")
ax.set_xlabel("Feature1")
ax.set_ylabel("Feature2")

st.pyplot(fig)


# Q5


st.header("Q5. Comparison of KMeans and DBSCAN")

st.markdown("""
### KMeans

- Assumes spherical clusters.
- Splits moon-shaped data incorrectly.
- Cannot detect noise.
- Not suitable for non-linear datasets.

### DBSCAN

- Detects non-linear clusters correctly.
- Finds clusters of arbitrary shape.
- Detects noise points (-1 label).
- Better suited for moon-shaped datasets.
""")


# Q6. DBSCAN Parameter Tuning

st.header("Q6. DBSCAN Parameter Tuning")

eps_values = [0.2, 0.3, 0.4, 0.5]

results = []

for eps in eps_values:

    model = DBSCAN(
        eps=eps,
        min_samples=5
    )

    labels = model.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_points = list(labels).count(-1)

    results.append([eps, n_clusters, noise_points])

result_df = pd.DataFrame(
    results,
    columns=["eps", "Number of Clusters", "Noise Points"]
)

st.write("### DBSCAN Results")
st.dataframe(result_df)

st.success("Best eps value for the moon dataset: 0.3")


# Q7. Creating High-Dimensional Dataset


st.header("Q7. Creating High-Dimensional Dataset")

X_blob, y_blob = make_blobs(
    n_samples=500,
    n_features=6,
    centers=4,
    cluster_std=1.5,
    random_state=42
)

scaler_blob = StandardScaler()

X_blob_scaled = scaler_blob.fit_transform(X_blob)

st.write("Dataset Shape:", X_blob.shape)

# Q8. PCA

st.header("Q8. Applying PCA")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_blob_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["Original Cluster"] = y_blob

st.write("### First 5 Rows")
st.dataframe(pca_df.head())

st.write("### Explained Variance Ratio")
st.write(pca.explained_variance_ratio_)

# Q9. PCA Visualization

st.header("Q9. PCA Visualization")

fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Original Cluster",
    palette="Set1",
    ax=ax
)

ax.set_title("PCA Visualization of High-Dimensional Dataset")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

st.pyplot(fig)

st.info(
    f"""
The first two principal components retain approximately
**{(pca.explained_variance_ratio_.sum()*100):.2f}%**
of the total variance (information) in the dataset.
"""
)


# Q10. Mini Project

st.header("Q10")

iris = load_iris()

iris_df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

st.write("### Original Dataset")
st.dataframe(iris_df.head())

# Scaling
scaler = StandardScaler()

iris_scaled = scaler.fit_transform(iris_df)

# DBSCAN
dbscan = DBSCAN(
    eps=0.8,
    min_samples=5
)

iris_clusters = dbscan.fit_predict(iris_scaled)

# PCA
pca = PCA(n_components=2)

iris_pca = pca.fit_transform(iris_scaled)

iris_plot = pd.DataFrame(
    iris_pca,
    columns=["PC1", "PC2"]
)

iris_plot["Cluster"] = iris_clusters

st.write("### PCA Dataset")
st.dataframe(iris_plot.head())

fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(
    data=iris_plot,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set2",
    ax=ax
)

ax.set_title("DBSCAN + PCA on Iris Dataset")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

st.pyplot(fig)



