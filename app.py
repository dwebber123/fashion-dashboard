import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Page Config ---
st.set_page_config(
    page_title="Fashion Retail Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling with Dark Mode and Background ---
custom_css = """
<style>
body {
    background-color: #0e1117;
    color: #ffffff;
}
[data-testid="stSidebar"] {
    background-color: #1e2127;
}
h1, h2, h3, .stTextInput > label, .stSelectbox > label {
    color: #ffffff !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Add Logo ---
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Fashion_Design_logo.png/240px-Fashion_Design_logo.png", width=120)


st.title("🛍️ Fashion Retail Customer Insights Dashboard")

# --- Load data ---
df = pd.read_csv("clothing_retail_300.csv", parse_dates=["InvoiceDate"])
rfm = pd.read_csv("rfm_with_clusters.csv")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Options")
country = st.sidebar.selectbox("Select Country", df["Country"].unique())
season = st.sidebar.selectbox("Select Season", df["Season"].unique())

filtered_df = df[(df["Country"] == country) & (df["Season"] == season)]
st.markdown(f"### Showing data for **{country}** during **{season}**")

# --- Cluster Selector ---
st.subheader("📊 Customer Segmentation")
selected_cluster = st.selectbox("Select a Customer Segment (Cluster)", sorted(rfm["Cluster"].unique()))
filtered_rfm = rfm[rfm["Cluster"] == selected_cluster]
st.dataframe(filtered_rfm)

cols = st.columns(3)
cols[0].metric("Avg. Recency", f"{filtered_rfm['Recency'].mean():.1f} days")
cols[1].metric("Avg. Frequency", f"{filtered_rfm['Frequency'].mean():.1f} purchases")
cols[2].metric("Avg. Monetary", f"${filtered_rfm['Monetary'].mean():,.2f}")

# --- PCA Visualization ---
st.subheader("🧭 PCA Cluster Projection")
scaler = StandardScaler()
X = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X)
pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"])
pca_df["Cluster"] = rfm["Cluster"]

fig1, ax1 = plt.subplots()
sns.set_theme(style="darkgrid")
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax1)
ax1.set_title("Customer Segments by PCA")
st.pyplot(fig1)

# --- Top Categories ---
st.subheader("🧺 Top Product Categories")
top_categories = filtered_df["Category"].value_counts().head(5)
st.bar_chart(top_categories)

# --- Size Curve ---
st.subheader("📏 Size Distribution")
size_counts = filtered_df["Size"].value_counts().reindex(["XS", "S", "M", "L", "XL"])
st.line_chart(size_counts)

# --- Gender Split ---
st.subheader("🚻 Gender Distribution")
gender_counts = filtered_df["Gender"].value_counts()
st.bar_chart(gender_counts)

# --- Seasonal Revenue ---
st.subheader("📅 Seasonal Revenue Breakdown")
seasonal_revenue = df.groupby("Season")["LineTotal"].sum().reindex(["Winter", "Spring", "Summer", "Fall"])
st.area_chart(seasonal_revenue)

# --- Association Rules Table ---
st.subheader("🤝 Association Rules")
try:
    rules = pd.read_csv("rules_sorted.csv")
    rules["antecedents"] = rules["antecedents"].str.replace("frozenset", "").str.strip("(){}")
    rules["consequents"] = rules["consequents"].str.replace("frozenset", "").str.strip("(){}")
    st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))
except FileNotFoundError:
    st.warning("No association rules file found. Please generate 'rules_sorted.csv' if needed.")

# --- Download Button ---
st.download_button("📥 Download RFM Segment", rfm.to_csv(index=False), "rfm_with_clusters.csv")

st.caption("Built with 🌟 Streamlit | Dataset: clothing_retail_300.csv")

# --- Deployment note ---
st.markdown("""
---
### 🌐 Deploy this on Streamlit Cloud
1. Push your project to GitHub.
2. Go to https://streamlit.io/cloud and sign in.
3. Connect your repo and select `app.py`.
4. Your dashboard will be live in seconds.
""")
