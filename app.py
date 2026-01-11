import streamlit as st
import pandas as pd

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🛒 Shopping Cart Customer Clusters")

clusters = pd.read_csv("data/processed/customer_clusters_final.csv")
rules = pd.read_csv("data/processed/rules_apriori_filtered.csv")

cluster_id = st.selectbox("Chọn cluster", sorted(clusters["cluster"].unique()))

st.subheader("📊 Thông tin khách hàng")
st.write(clusters[clusters["cluster"] == cluster_id].describe())

st.subheader("🔗 Top Rules")
st.write(rules.head(10))
