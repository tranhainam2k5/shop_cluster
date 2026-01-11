# ============================================================
# MINI PROJECT: RULE-BASED TRANSACTION CLUSTERING
# ============================================================

import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ============================================================
# 1. LOAD DATA
# ============================================================

print("🔹 1. Loading data...")

TRANSACTION_DATA_PATH = "data/processed/cleaned_uk_data.csv"
RULES_PATH = "data/processed/rules_apriori_filtered.csv"

assert os.path.exists(TRANSACTION_DATA_PATH), "❌ cleaned_uk_data.csv not found"
assert os.path.exists(RULES_PATH), "❌ rules_apriori_filtered.csv not found"

df_raw = pd.read_csv(TRANSACTION_DATA_PATH, encoding="ISO-8859-1")
rules_df = pd.read_csv(RULES_PATH)

print(f"✓ Raw data loaded: {df_raw.shape}")
print(f"✓ Rules loaded: {rules_df.shape}")


# ============================================================
# 2. CREATE TransactionID – Item DATA
# ============================================================

print("\n🔹 2. Creating TransactionID – Item data...")

df_transactions = (
    df_raw
    .groupby(["InvoiceNo", "Description"])
    .size()
    .reset_index(name="count")
    .rename(columns={
        "InvoiceNo": "TransactionID",
        "Description": "Item"
    })
)

print(f"✓ Transactions created: {df_transactions.shape}")
print(df_transactions.head())


# ============================================================
# 3. RULE-BASED FEATURE ENGINEERING
# ============================================================

print("\n🔹 3. Rule → Feature Engineering...")

# Parse antecedents & consequents
def parse_frozenset(x):
    """
    Parse string dạng frozenset({'A', 'B'}) thành Python set
    """
    if isinstance(x, str):
        x = x.replace("frozenset", "")
        return set(ast.literal_eval(x))
    return set(x)


rules_df["antecedents"] = rules_df["antecedents"].apply(parse_frozenset)
rules_df["consequents"] = rules_df["consequents"].apply(parse_frozenset)


# Group items per transaction
transaction_items = (
    df_transactions
    .groupby("TransactionID")["Item"]
    .apply(set)
)

feature_matrix = []

for _, rule in rules_df.iterrows():
    antecedent = rule["antecedents"]
    consequent = rule["consequents"]

    feature_matrix.append(
        transaction_items.apply(
            lambda items: int(
                antecedent.issubset(items) and
                consequent.issubset(items)
            )
        )
    )

rule_feature_df = pd.DataFrame(feature_matrix).T
rule_feature_df.columns = [f"Rule_{i}" for i in range(len(rules_df))]
rule_feature_df.index.name = "TransactionID"

print(f"✓ Rule-feature matrix shape: {rule_feature_df.shape}")
print("\n🔹 3B. Creating weighted rule features (lift × confidence)...")

weighted_feature_matrix = []

for _, rule in rules_df.iterrows():
    antecedent = rule["antecedents"]
    consequent = rule["consequents"]
    weight = rule["lift"] * rule["confidence"]

    weighted_feature_matrix.append(
        transaction_items.apply(
            lambda items: weight if (
                antecedent.issubset(items) and
                consequent.issubset(items)
            ) else 0
        )
    )

rule_feature_weighted_df = pd.DataFrame(weighted_feature_matrix).T
rule_feature_weighted_df.columns = [
    f"RuleW_{i}" for i in range(len(rules_df))
]
rule_feature_weighted_df.index = rule_feature_df.index

print(f"✓ Weighted rule-feature matrix shape: {rule_feature_weighted_df.shape}")


# ============================================================
# 4. STANDARDIZATION
# ============================================================

print("\n🔹 4. Standardizing features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rule_feature_df)

X = pd.DataFrame(
    X_scaled,
    index=rule_feature_df.index,
    columns=rule_feature_df.columns
)

print("✓ Data standardized")


# ============================================================
# 5. SELECT NUMBER OF CLUSTERS (SILHOUETTE)
# ============================================================

print("\n🔹 5. Selecting number of clusters (K)...")

sil_scores = {}

for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores[k] = score
    print(f"  - K={k}: Silhouette Score = {score:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"✓ Selected K = {best_k}")
print("\n🔹 5B. Comparing Binary vs Weighted Rule Features...")

def evaluate_feature_variant(X, name):
    scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k = max(scores, key=scores.get)
    print(f"  {name}: Best K={best_k}, Silhouette={scores[best_k]:.4f}")
    return best_k, scores[best_k]


# Binary
best_k_bin, sil_bin = evaluate_feature_variant(X, "Binary Rules")

# Weighted
scaler_w = StandardScaler()
Xw_scaled = scaler_w.fit_transform(rule_feature_weighted_df)
Xw = pd.DataFrame(
    Xw_scaled,
    index=rule_feature_weighted_df.index,
    columns=rule_feature_weighted_df.columns
)

best_k_w, sil_w = evaluate_feature_variant(Xw, "Weighted Rules")

print("\n📊 Feature Comparison Summary")
print(pd.DataFrame({
    "Feature": ["Binary Rules", "Weighted Rules"],
    "Best_K": [best_k_bin, best_k_w],
    "Silhouette": [sil_bin, sil_w]
}))



# ============================================================
# 6. K-MEANS CLUSTERING
# ============================================================

print("\n🔹 6. Training K-Means...")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

clustered_df = rule_feature_df.copy()
clustered_df["cluster"] = cluster_labels

print("✓ Clustering completed")
print(clustered_df["cluster"].value_counts())


# ============================================================
# 7. PCA & VISUALIZATION
# ============================================================

print("\n🔹 7. PCA & Visualization...")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"],
    index=X.index
)
pca_df["cluster"] = cluster_labels

plt.figure(figsize=(8, 6))
for c in sorted(pca_df["cluster"].unique()):
    subset = pca_df[pca_df["cluster"] == c]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Cluster {c}",
        alpha=0.7
    )

plt.title("PCA Visualization of Rule-Based Transaction Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 8. CLUSTER PROFILING & MARKETING INSIGHT
# ============================================================

print("\n🔹 8. Cluster Profiling & Marketing Insight")

profiles = []

for c in sorted(clustered_df["cluster"].unique()):
    group = clustered_df[clustered_df["cluster"] == c]

    top_rules = (
        group
        .drop(columns="cluster")
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .index
        .tolist()
    )

    profile = {
        "cluster": c,
        "n_transactions": len(group),
        "top_rules": top_rules
    }

    # Naming & strategy
    if len(group) > 100000:
        profile["name_en"] = "Heavy Buyers"
        profile["name_vi"] = "Khách mua nhiều"
        profile["strategy"] = "Cross-sell & bundle các sản phẩm mua kèm"
    elif len(group) > 50000:
        profile["name_en"] = "Regular Buyers"
        profile["name_vi"] = "Khách mua đều"
        profile["strategy"] = "Ưu đãi theo nhóm sản phẩm quen thuộc"
    else:
        profile["name_en"] = "Occasional Buyers"
        profile["name_vi"] = "Khách mua không thường xuyên"
        profile["strategy"] = "Kích hoạt bằng khuyến mãi & email remarketing"

    profiles.append(profile)

profile_df = pd.DataFrame(profiles)

print("\n📊 Cluster Profiles & Marketing Strategy:")
print(profile_df)

profile_df.to_csv(
    "data/processed/cluster_marketing_profiles.csv",
    index=False
)
# =========================================================
# 9. CLUSTER PROFILING & INTERPRETATION
# =========================================================

def profile_clusters(
    meta_df: pd.DataFrame,
    labels: np.ndarray,
    rule_features: np.ndarray,
    rules_df: pd.DataFrame,
    top_n_rules: int = 10,
):
    """
    Profiling từng cluster:
    - số khách
    - RFM trung bình
    - top rule features
    """

    meta = meta_df.copy()
    meta["cluster"] = labels

    summaries = []

    for c in sorted(meta["cluster"].unique()):
        group = meta[meta["cluster"] == c]

        summary = {
            "cluster": c,
            "n_customers": len(group),
        }

        for col in ["Recency", "Frequency", "Monetary"]:
            if col in group.columns:
                summary[f"{col}_mean"] = group[col].mean()

        rule_strength = rule_features[meta["cluster"] == c].mean(axis=0)
        top_idx = np.argsort(rule_strength)[::-1][:top_n_rules]

        summary["top_rules"] = rules_df.loc[top_idx, "rule_str"].tolist()

        summaries.append(summary)

    return summaries
# =========================================================
# 10. CLUSTER NAMING & MARKETING STRATEGY
# =========================================================

def assign_marketing_strategy(cluster_profiles: list[dict]):
    """
    Gán tên, persona và chiến lược marketing cho từng cluster
    """

    strategies = []

    for p in cluster_profiles:
        c = p["cluster"]

        if p.get("Monetary_mean", 0) > 5000:
            name_en = "High Value Loyalists"
            name_vi = "Khách hàng trung thành giá trị cao"
            strategy = "Upsell sản phẩm cao cấp, chương trình VIP"
        elif p.get("Frequency_mean", 0) > 20:
            name_en = "Frequent Shoppers"
            name_vi = "Khách mua thường xuyên"
            strategy = "Bundle sản phẩm mua kèm, cross-sell"
        else:
            name_en = "Occasional Buyers"
            name_vi = "Khách mua không thường xuyên"
            strategy = "Khuyến mãi kích hoạt, email remarketing"

        strategies.append({
            "cluster": c,
            "name_en": name_en,
            "name_vi": name_vi,
            "persona": f"Cluster {c} gồm {p['n_customers']} khách với hành vi đặc trưng.",
            "strategy": strategy,
            "top_rules": p["top_rules"]
        })

    return pd.DataFrame(strategies)

