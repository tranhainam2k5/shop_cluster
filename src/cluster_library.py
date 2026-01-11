# -*- coding: utf-8 -*-
"""
Shopping Cart Library

This library contains classes for data cleaning, feature engineering,
and association rule analysis for shopping cart.
"""

import datetime as dt
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, TruncatedSVD
import plotly.express as px
import networkx as nx


# =========================================================
# 1. DATA CLEANER
# =========================================================

class DataCleaner:
    """
    A class for cleaning and preprocessing retail transaction data.

    This class handles data loading, cleaning operations, and basic exploratory
    data analysis for online retail datasets.
    """

    def __init__(self, data_path):
        """
        Initialize the DataCleaner with data path.

        Args:
            data_path (str): Path to the raw data file
        """
        self.data_path = data_path
        self.df = None
        self.df_uk = None
        self.rfm_data = None

    def load_data(self):
        """
        Load and display basic information about the dataset.

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        dtype = dict(
            InvoiceNo=np.object_,
            StockCode=np.object_,
            Description=np.object_,
            Quantity=np.int64,
            UnitPrice=np.float64,
            CustomerID=np.object_,
            Country=np.object_,
        )

        self.df = pd.read_csv(
            self.data_path,
            encoding="ISO-8859-1",
            parse_dates=["InvoiceDate"],
            dtype=dtype,
        )

        # Chuyển CustomerID thành format 6 ký tự
        self.df["CustomerID"] = (
            self.df["CustomerID"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(6)
        )

        print(f"Kích thước dữ liệu: {self.df.shape}")
        print(f"Số bản ghi: {len(self.df):,}")

        return self.df

    def clean_data(self):
        """
        Clean the dataset by removing invalid records and focusing on UK customers.

        Returns:
            pd.DataFrame: Cleaned UK dataset
        """
        if self.df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        # Thêm cột TotalPrice
        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]

        # Loại bỏ các hóa đơn bị hủy (bắt đầu bằng 'C')
        self.df = self.df[~self.df["InvoiceNo"].astype(str).str.startswith("C")]

        # Chỉ tập trung vào khách hàng UK
        self.df_uk = self.df[self.df["Country"] == "United Kingdom"].copy()

        # Loại bỏ các sản phẩm có quantity hoặc price không hợp lệ
        self.df_uk = self.df_uk[
            (self.df_uk["Quantity"] > 0) & (self.df_uk["UnitPrice"] > 0)
        ]

        # Bỏ description NA
        self.df_uk = self.df_uk.dropna(subset=["Description"])

        return self.df_uk

    def create_time_features(self):
        """
        Create time-based features for analysis.
        """
        if self.df_uk is None:
            raise ValueError("Cleaned UK data not available. Call clean_data() first.")

        self.df_uk["DayOfWeek"] = self.df_uk["InvoiceDate"].dt.dayofweek
        self.df_uk["HourOfDay"] = self.df_uk["InvoiceDate"].dt.hour

    def add_total_price(self):
        """
        Add TotalPrice column (Quantity * UnitPrice) to cleaned UK data.
        """
        if self.df_uk is None:
            raise ValueError("Cleaned UK data not available. Call clean_data() first.")

        self.df_uk["TotalPrice"] = self.df_uk["Quantity"] * self.df_uk["UnitPrice"]
        return self.df_uk

    def compute_rfm(self, snapshot_date=None):
        """
        Compute RFM (Recency, Frequency, Monetary) for each customer based on cleaned UK data.

        Args:
            snapshot_date (datetime or str, optional):
                Reference date for Recency calculation.
                - If None: use max(InvoiceDate) + 1 day.

        Returns:
            pd.DataFrame: RFM dataframe with columns [CustomerID, Recency, Frequency, Monetary]
        """
        if self.df_uk is None:
            raise ValueError("Cleaned UK data not available. Call clean_data() first.")

        df = self.df_uk.copy()

        # Đảm bảo có TotalPrice
        if "TotalPrice" not in df.columns:
            df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

        # Xác định snapshot_date
        if snapshot_date is None:
            snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        else:
            # Cho phép truyền vào dạng string 'YYYY-MM-DD'
            if isinstance(snapshot_date, str):
                snapshot_date = pd.to_datetime(snapshot_date)

        # Tính RFM
        rfm = df.groupby("CustomerID").agg(
            {
                "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # Recency
                "InvoiceNo": "nunique",  # Frequency
                "TotalPrice": "sum",     # Monetary
            }
        )

        rfm.rename(
            columns={
                "InvoiceDate": "Recency",
                "InvoiceNo": "Frequency",
                "TotalPrice": "Monetary",
            },
            inplace=True,
        )

        self.rfm_data = rfm.reset_index()
        return self.rfm_data

    def save_cleaned_data(self, output_dir="../data/processed"):
        """
        Save cleaned data to specified directory.

        Args:
            output_dir (str): Output directory path
        """
        if self.df_uk is None:
            raise ValueError("Cleaned UK data not available. Call clean_data() first.")

        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/cleaned_uk_data.csv"
        self.df_uk.to_csv(output_path, index=False)
        print(f"Đã lưu dữ liệu đã làm sạch: {output_path}")


# =========================================================
# 2. BASKET PREPARER
# =========================================================

class BasketPreparer:
    """
    A class for preparing basket data for association rule mining.

    This class transforms transaction data into a format suitable for
    applying the Apriori algorithm.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        invoice_col: str = "InvoiceNo",
        item_col: str = "Description",
        quantity_col: str = "Quantity",
    ):
        """
        Initialize the BasketPreparer with cleaned dataframe.

        Args:
            df (pd.DataFrame): Cleaned transaction-level dataframe
            invoice_col (str): Column name for invoice number
            item_col (str): Column name for item description
            quantity_col (str): Column name for item quantity
        """
        self.df = df
        self.invoice_col = invoice_col
        self.item_col = item_col
        self.quantity_col = quantity_col
        self.basket = None
        self.basket_bool = None

    def create_basket(self):
        """
        Create a basket format dataframe for Apriori algorithm.

        Returns:
            pd.DataFrame: Basket format dataframe
        """

        basket = (
            self.df.groupby([self.invoice_col, self.item_col])[self.quantity_col]
            .sum()
            .unstack()
            .fillna(0)
        )

        self.basket = basket
        return self.basket

    def encode_basket(self, threshold: int = 1):
        """
        Encode the basket dataframe into boolean format.

        Args:
            threshold (int): Minimum quantity to consider an item as present

        Returns:
            pd.DataFrame: Boolean encoded basket dataframe
        """

        if self.basket is None:
            raise ValueError("Basket not created. Please run create_basket() first.")
        basket_bool = self.basket.applymap(lambda x: 1 if x >= threshold else 0)
        basket_bool = basket_bool.astype(bool)
        self.basket_bool = basket_bool
        return self.basket_bool

    def save_basket_bool(self, output_path: str):
        """
        Save the boolean encoded basket dataframe to a Parquet file.

        Args:
            output_path (str): Path to save the Parquet file
        """
        if self.basket_bool is None:
            raise ValueError("Basket not encoded. Please call encode_basket() first.")
        basket_bool_to_save = self.basket_bool.reset_index(drop=True)

        basket_bool_to_save.to_parquet(output_path, index=False)
        print(f"Đã lưu basket boolean: {output_path}")


# =========================================================
# 3. APRIORI ASSOCIATION RULES MINER
# =========================================================

class AssociationRulesMiner:
    """
    A class for mining association rules using the Apriori algorithm.

    This class applies the Apriori algorithm to the basket data and extracts
    association rules based on specified metrics.
    """

    def __init__(self, basket_bool: pd.DataFrame):
        """
        Initialize the AssociationRulesMiner with basket data.

        Args:
            basket_bool (pd.DataFrame): Boolean encoded basket dataframe
        """
        self.basket_bool = basket_bool
        self.frequent_itemsets = None
        self.rules = None

    def mine_frequent_itemsets(
        self,
        min_support: float = 0.01,
        max_len: int = None,
        use_colnames: bool = True,
    ) -> pd.DataFrame:
        """
        Mine frequent itemsets using the Apriori algorithm.

        Returns:
            pd.DataFrame: DataFrame of frequent itemsets
        """

        fi = apriori(
            self.basket_bool,
            min_support=min_support,
            use_colnames=use_colnames,
            max_len=max_len,
        )

        fi.sort_values(by="support", ascending=False, inplace=True)
        self.frequent_itemsets = fi
        return self.frequent_itemsets

    def generate_rules(
        self,
        metric: str = "lift",
        min_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets.

        Args:
            metric (str): Metric to evaluate the rules
            min_threshold (float): Minimum threshold for the metric

        Returns:
            pd.DataFrame: DataFrame of association rules
        """

        if self.frequent_itemsets is None:
            raise ValueError(
                "Frequent itemsets not mined. Please run mine_frequent_itemsets() first."
            )

        rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold,
        )

        rules = rules.sort_values(["lift", "confidence"], ascending=False)
        self.rules = rules
        return self.rules

    @staticmethod
    def _frozenset_to_str(fs: frozenset) -> str:
        return ", ".join(sorted(list(fs)))

    def add_readable_rule_str(self) -> pd.DataFrame:
        """
        Add human-readable columns for antecedents, consequents, and rule_str
        to the rules dataframe.

        Returns:
            pd.DataFrame: Rules dataframe with extra readable columns
        """
        if self.rules is None:
            raise ValueError("rules is not available. Call generate_rules() first.")

        rules = self.rules.copy()
        rules["antecedents_str"] = rules["antecedents"].apply(self._frozenset_to_str)
        rules["consequents_str"] = rules["consequents"].apply(self._frozenset_to_str)
        rules["rule_str"] = rules["antecedents_str"] + " → " + rules["consequents_str"]

        self.rules = rules
        return self.rules

    def filter_rules(
        self,
        min_support: float = None,
        min_confidence: float = None,
        min_lift: float = None,
        max_len_antecedents: int = None,
        max_len_consequents: int = None,
    ) -> pd.DataFrame:
        """
        Filter rules based on support, confidence, lift and length of antecedents/consequents.
        """
        if self.rules is None:
            raise ValueError("rules is not available. Call generate_rules() first.")

        filtered = self.rules.copy()

        if min_support is not None:
            filtered = filtered[filtered["support"] >= min_support]
        if min_confidence is not None:
            filtered = filtered[filtered["confidence"] >= min_confidence]
        if min_lift is not None:
            filtered = filtered[filtered["lift"] >= min_lift]
        if max_len_antecedents is not None:
            filtered = filtered[
                filtered["antecedents"].apply(len) <= max_len_antecedents
            ]
        if max_len_consequents is not None:
            filtered = filtered[
                filtered["consequents"].apply(len) <= max_len_consequents
            ]

        filtered = filtered.reset_index(drop=True)
        return filtered

    def save_rules(self, output_path: str, rules_df: pd.DataFrame = None):
        """
        Save rules dataframe to CSV.

        Args:
            output_path (str): CSV path
            rules_df (pd.DataFrame): Rules dataframe to save (if None, use self.rules)
        """
        if rules_df is None:
            if self.rules is None:
                raise ValueError("No rules to save.")
            rules_df = self.rules

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rules_df.to_csv(output_path, index=False)
        print(f"Đã lưu luật vào: {output_path}")

# =========================================================
# 4. FP-GROWTH ASSOCIATION RULES MINER
# =========================================================


class FPGrowthMiner:
    """
    A class for mining association rules using the FP-Growth algorithm.

    Interface được thiết kế tương tự AssociationRulesMiner (Apriori)
    để dễ tái sử dụng và so sánh.
    """

    def __init__(self, basket_bool: pd.DataFrame):
        """
        Initialize the FPGrowthMiner with basket data.

        Args:
            basket_bool (pd.DataFrame): Boolean encoded basket dataframe
        """
        self.basket_bool = basket_bool
        self.frequent_itemsets = None
        self.rules = None

    def mine_frequent_itemsets(
        self,
        min_support: float = 0.01,
        max_len: int = None,
        use_colnames: bool = True,
    ) -> pd.DataFrame:
        """
        Mine frequent itemsets using the FP-Growth algorithm.

        Args:
            min_support (float): Ngưỡng support tối thiểu.
            max_len (int | None): Độ dài tối đa của itemset.
            use_colnames (bool): True nếu muốn itemsets dùng tên cột.

        Returns:
            pd.DataFrame: DataFrame of frequent itemsets
        """
        fi = fpgrowth(
            self.basket_bool,
            min_support=min_support,
            use_colnames=use_colnames,
            max_len=max_len,
        )
        fi.sort_values(by="support", ascending=False, inplace=True)
        self.frequent_itemsets = fi
        return self.frequent_itemsets

    def generate_rules(
        self,
        metric: str = "lift",
        min_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets.

        Args:
            metric (str): Metric to evaluate the rules
            min_threshold (float): Minimum threshold for the metric

        Returns:
            pd.DataFrame: DataFrame of association rules
        """
        if self.frequent_itemsets is None:
            raise ValueError(
                "Frequent itemsets not mined. "
                "Please run mine_frequent_itemsets() first."
            )

        rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold,
        )
        rules = rules.sort_values(["lift", "confidence"], ascending=False)
        self.rules = rules
        return self.rules

    @staticmethod
    def _frozenset_to_str(fs: frozenset) -> str:
        return ", ".join(sorted(list(fs)))

    def add_readable_rule_str(self) -> pd.DataFrame:
        """
        Add human-readable columns for antecedents, consequents, and rule_str
        to the rules dataframe.

        Returns:
            pd.DataFrame: Rules dataframe with extra readable columns
        """
        if self.rules is None:
            raise ValueError("rules is not available. Call generate_rules() first.")

        rules = self.rules.copy()
        rules["antecedents_str"] = rules["antecedents"].apply(self._frozenset_to_str)
        rules["consequents_str"] = rules["consequents"].apply(self._frozenset_to_str)
        rules["rule_str"] = rules["antecedents_str"] + " → " + rules["consequents_str"]
        self.rules = rules
        return self.rules

    def filter_rules(
        self,
        min_support: float = None,
        min_confidence: float = None,
        min_lift: float = None,
        max_len_antecedents: int = None,
        max_len_consequents: int = None,
    ) -> pd.DataFrame:
        """
        Filter rules based on support, confidence, lift and length of
        antecedents/consequents.
        """
        if self.rules is None:
            raise ValueError("rules is not available. Call generate_rules() first.")

        filtered = self.rules.copy()

        if min_support is not None:
            filtered = filtered[filtered["support"] >= min_support]
        if min_confidence is not None:
            filtered = filtered[filtered["confidence"] >= min_confidence]
        if min_lift is not None:
            filtered = filtered[filtered["lift"] >= min_lift]

        if max_len_antecedents is not None:
            filtered = filtered[
                filtered["antecedents"].apply(len) <= max_len_antecedents
            ]
        if max_len_consequents is not None:
            filtered = filtered[
                filtered["consequents"].apply(len) <= max_len_consequents
            ]

        filtered = filtered.reset_index(drop=True)
        return filtered

    def save_rules(self, output_path: str, rules_df: pd.DataFrame = None):
        """
        Save rules dataframe to CSV.

        Args:
            output_path (str): CSV path
            rules_df (pd.DataFrame): Rules dataframe to save
                (if None, use self.rules)
        """
        if rules_df is None:
            if self.rules is None:
                raise ValueError("No rules to save.")
            rules_df = self.rules

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rules_df.to_csv(output_path, index=False)
        print(f"Đã lưu luật vào: {output_path}")

# =========================================================
# 5. APRIORI vs FP-GROWTH COMPARISON HELPERS
# =========================================================


def benchmark_apriori_vs_fpgrowth(
    basket_bool: pd.DataFrame,
    min_support: float = 0.01,
    max_len: int = None,
    metric: str = "lift",
    min_threshold: float = 1.0,
) -> dict:
    """
    Chạy cả Apriori và FP-Growth trên cùng một basket_bool, đo thời gian
    và trả về summary để phục vụ so sánh trong notebook.

    Returns:
        dict với các keys:
            - "summary": pd.DataFrame với các cột:
                ['algorithm', 'runtime_sec', 'n_itemsets',
                 'n_rules', 'avg_itemset_length']
            - "apriori_itemsets": frequent itemsets từ Apriori
            - "apriori_rules": rules từ Apriori
            - "fpgrowth_itemsets": frequent itemsets từ FP-Growth
            - "fpgrowth_rules": rules từ FP-Growth
    """
    # --- Apriori ---
    apriori_miner = AssociationRulesMiner(basket_bool=basket_bool)
    t0 = time.time()
    fi_ap = apriori_miner.mine_frequent_itemsets(
        min_support=min_support,
        max_len=max_len,
        use_colnames=True,
    )
    rules_ap = apriori_miner.generate_rules(metric=metric, min_threshold=min_threshold)
    t_ap = time.time() - t0

    avg_len_ap = (
        fi_ap["itemsets"].apply(len).mean() if not fi_ap.empty else 0.0
    )

    # --- FP-Growth ---
    fpg_miner = FPGrowthMiner(basket_bool=basket_bool)
    t0 = time.time()
    fi_fp = fpg_miner.mine_frequent_itemsets(
        min_support=min_support,
        max_len=max_len,
        use_colnames=True,
    )
    rules_fp = fpg_miner.generate_rules(metric=metric, min_threshold=min_threshold)
    t_fp = time.time() - t0

    avg_len_fp = (
        fi_fp["itemsets"].apply(len).mean() if not fi_fp.empty else 0.0
    )

    summary = pd.DataFrame(
        {
            "algorithm": ["apriori", "fpgrowth"],
            "runtime_sec": [t_ap, t_fp],
            "n_itemsets": [len(fi_ap), len(fi_fp)],
            "n_rules": [len(rules_ap), len(rules_fp)],
            "avg_itemset_length": [avg_len_ap, avg_len_fp],
        }
    )

    return {
        "summary": summary,
        "apriori_itemsets": fi_ap,
        "apriori_rules": rules_ap,
        "fpgrowth_itemsets": fi_fp,
        "fpgrowth_rules": rules_fp,
    }


# =========================================================
# 6. DATA VISUALIZER (EDA + RFM + ASSOCIATION RULES)
# =========================================================

class DataVisualizer:
    """
    A class for creating visualizations for customer segmentation and
    shopping behavior analysis.

    This class provides methods for plotting various aspects of the data
    including temporal patterns, customer behavior, RFM analysis,
    và trực quan hoá luật kết hợp (Apriori).
    """

    def __init__(self):
        """Initialize the DataVisualizer with plotting settings."""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("viridis")

    def plot_revenue_over_time(self, df):
        """
        Plot daily and monthly revenue patterns.

        Args:
            df (pd.DataFrame): Dataframe with InvoiceDate and TotalPrice columns
        """
        # Daily revenue
        plt.figure(figsize=(12, 5))
        daily_revenue = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        daily_revenue.plot()
        plt.title("Doanh thu hàng ngày")
        plt.xlabel("Ngày")
        plt.ylabel("Doanh thu (GBP)")
        plt.tight_layout()
        plt.show()

        # Monthly revenue
        plt.figure(figsize=(12, 5))
        monthly_revenue = df.groupby(pd.Grouper(key="InvoiceDate", freq="M"))[
            "TotalPrice"
        ].sum()
        monthly_revenue.plot(kind="bar")
        plt.title("Doanh thu hàng tháng")
        plt.xlabel("Tháng")
        plt.ylabel("Doanh thu (GBP)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_time_patterns(self, df):
        """
        Plot purchase patterns by day and hour.

        Args:
            df (pd.DataFrame): Dataframe with time features:
                DayOfWeek, HourOfDay
        """
        plt.figure(figsize=(12, 5))
        day_hour_counts = (
            df.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)
        )
        sns.heatmap(day_hour_counts, cmap="viridis")
        plt.title("Hoạt động mua hàng theo ngày và giờ")
        plt.xlabel("Giờ trong ngày")
        plt.ylabel("Ngày trong tuần (0=Thứ 2, 6=Chủ nhật)")
        plt.tight_layout()
        plt.show()

    def plot_product_analysis(self, df, top_n=10):
        """
        Plot top products by quantity and revenue.

        Args:
            df (pd.DataFrame): Transaction dataframe (có Quantity, TotalPrice)
            top_n (int): Number of top products to show
        """
        # Top sản phẩm theo số lượng
        plt.figure(figsize=(12, 5))
        top_products = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_products.values, y=top_products.index)
        plt.title(f"Top {top_n} sản phẩm theo số lượng bán")
        plt.xlabel("Số lượng bán")
        plt.tight_layout()
        plt.show()

        # Top sản phẩm theo doanh thu
        plt.figure(figsize=(12, 5))
        top_revenue_products = (
            df.groupby("Description")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_revenue_products.values, y=top_revenue_products.index)
        plt.title(f"Top {top_n} sản phẩm theo doanh thu")
        plt.xlabel("Doanh thu (GBP)")
        plt.tight_layout()
        plt.show()

    def plot_customer_distribution(self, df):
        """
        Plot customer behavior distributions.

        Args:
            df (pd.DataFrame): Transaction dataframe with CustomerID, InvoiceNo, TotalPrice
        """
        # Số giao dịch trên mỗi khách hàng
        plt.figure(figsize=(10, 5))
        transactions_per_customer = df.groupby("CustomerID")["InvoiceNo"].nunique()
        sns.histplot(transactions_per_customer, bins=30, kde=True)
        plt.title("Phân phối số giao dịch trên mỗi khách hàng")
        plt.xlabel("Số giao dịch")
        plt.ylabel("Số khách hàng")
        plt.tight_layout()
        plt.show()

        # Chi tiêu trên mỗi khách hàng
        plt.figure(figsize=(10, 5))
        spend_per_customer = df.groupby("CustomerID")["TotalPrice"].sum()
        spend_filter = spend_per_customer < spend_per_customer.quantile(0.99)
        sns.histplot(spend_per_customer[spend_filter], bins=30, kde=True)
        plt.title("Phân phối tổng chi tiêu trên mỗi khách hàng")
        plt.xlabel("Tổng chi tiêu (GBP)")
        plt.ylabel("Số khách hàng")
        plt.tight_layout()
        plt.show()

    def plot_rfm_analysis(self, rfm_data):
        """
        Plot RFM analysis visualizations.

        Args:
            rfm_data (pd.DataFrame): RFM dataframe with
                columns ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        """
        # RFM distributions
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        sns.histplot(rfm_data["Recency"], bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Phân phối Recency (Ngày kể từ lần mua cuối)")
        axes[0].set_xlabel("Ngày")

        sns.histplot(rfm_data["Frequency"], bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Phân phối Frequency (Số giao dịch)")
        axes[1].set_xlabel("Số giao dịch")

        monetary_filter = rfm_data["Monetary"] < rfm_data["Monetary"].quantile(0.99)
        sns.histplot(
            rfm_data.loc[monetary_filter, "Monetary"], bins=30, kde=True, ax=axes[2]
        )
        axes[2].set_title("Phân phối Monetary (Tổng chi tiêu)")
        axes[2].set_xlabel("Tổng chi tiêu (GBP)")

        plt.tight_layout()
        plt.show()

# Apriori visualizations

    @staticmethod
    def _itemset_to_str(itemset):
        """
        Chuyển một itemset (frozenset, set, list, tuple) thành chuỗi có thể đọc được.

        Args:
            itemset: tập mục dưới dạng tập, danh sách, frozenset, v.v.
        """
        if isinstance(itemset, (set, frozenset, list, tuple)):
            return ", ".join(sorted(map(str, itemset)))
        return str(itemset)

    def plot_top_frequent_itemsets(
        self,
        frequent_itemsets: pd.DataFrame,
        top_n: int = 20,
        min_len: int | None = None,
        max_len: int | None = None,
        title: str = "Top frequent itemsets theo support",
    ):
        """
        Vẽ biểu đồ cột thể hiện các tập mục phổ biến nhất theo support.

        Args:
            frequent_itemsets: DataFrame kết quả từ mlxtend.frequent_patterns.apriori
                với tối thiểu hai cột 'itemsets' và 'support'.
            top_n: số lượng itemset hiển thị.
            min_len: chỉ lấy các itemset có độ dài >= min_len (nếu không None).
            max_len: chỉ lấy các itemset có độ dài <= max_len (nếu không None).
            title: tiêu đề biểu đồ.
        """
        if "itemsets" not in frequent_itemsets.columns or "support" not in frequent_itemsets.columns:
            raise ValueError("frequent_itemsets cần có cột 'itemsets' và 'support'.")

        fi = frequent_itemsets.copy()

        if min_len is not None:
            fi = fi[fi["itemsets"].apply(len) >= min_len]
        if max_len is not None:
            fi = fi[fi["itemsets"].apply(len) <= max_len]

        fi = fi.sort_values("support", ascending=False).head(top_n).copy()
        if fi.empty:
            print("Không có itemset nào thỏa mãn điều kiện lọc.")
            return

        fi["itemset_str"] = fi["itemsets"].apply(self._itemset_to_str)

        plt.figure(figsize=(12, max(4, 0.4 * len(fi))))
        sns.barplot(data=fi, x="support", y="itemset_str")
        plt.title(title)
        plt.xlabel("Support")
        plt.ylabel("Itemset")
        plt.tight_layout()
        plt.show()

    def plot_itemset_length_distribution(
        self,
        frequent_itemsets: pd.DataFrame,
        title: str = "Phân phối độ dài các tập mục (itemset length)",
    ):
        """
        Vẽ phân phối số lượng itemset theo độ dài (1-itemset, 2-itemset, ...).

        Args:
            frequent_itemsets: DataFrame kết quả từ apriori() với cột 'itemsets'.
            title: tiêu đề biểu đồ.
        """
        if "itemsets" not in frequent_itemsets.columns:
            raise ValueError("frequent_itemsets cần có cột 'itemsets'.")

        lengths = frequent_itemsets["itemsets"].apply(len)
        length_counts = lengths.value_counts().sort_index()

        plt.figure(figsize=(8, 5))
        sns.barplot(x=length_counts.index, y=length_counts.values)
        plt.title(title)
        plt.xlabel("Độ dài itemset")
        plt.ylabel("Số lượng itemset")
        plt.xticks(length_counts.index)
        plt.tight_layout()
        plt.show()

    def plot_top_rules_bar(
        self,
        rules_df: pd.DataFrame,
        top_n: int = 20,
        sort_by: str = "lift",
        title: str = "Top luật kết hợp",
    ):
        """
        Vẽ biểu đồ cột thể hiện top_n luật kết hợp theo một metric (lift/confidence/support).

        Args:
            rules_df: DataFrame kết quả từ association_rules() và đã có cột 'rule_str'.
            top_n: số luật hiển thị.
            sort_by: cột dùng để sắp xếp ('lift', 'confidence', 'support', ...).
            title: tiêu đề chung của biểu đồ.
        """
        if "rule_str" not in rules_df.columns:
            raise ValueError("rules_df cần có cột 'rule_str' (gọi add_readable_rule_str() trước).")
        if sort_by not in rules_df.columns:
            raise ValueError(f"rules_df không có cột '{sort_by}' để sắp xếp.")

        df = rules_df.sort_values(sort_by, ascending=False).head(top_n).copy()
        if df.empty:
            print("Không có luật nào để vẽ.")
            return

        plt.figure(figsize=(12, max(4, 0.4 * len(df))))
        sns.barplot(data=df, x=sort_by, y="rule_str")
        plt.title(f"{title} (theo {sort_by}) - Top {len(df)} luật")
        plt.xlabel(sort_by.capitalize())
        plt.ylabel("Luật (antecedent → consequent)")
        plt.tight_layout()
        plt.show()

    def plot_top_rules_lift(
        self,
        rules_df: pd.DataFrame,
        top_n: int = 20,
        title_prefix: str = "Top luật theo Lift (Apriori)",
    ):
        """
        Vẽ biểu đồ top luật theo chỉ số Lift.

        Args:
            rules_df: DataFrame, thường là rules_filtered_ap.
            top_n: số luật lấy top theo lift.
            title_prefix: phần tiêu đề, sẽ gắn thêm số luật thực tế.
        """
        if rules_df is None or rules_df.empty:
            print("Không có luật nào sau khi lọc để vẽ top lift.")
            return

        self.plot_top_rules_bar(
            rules_df=rules_df,
            top_n=top_n,
            sort_by="lift",
            title=title_prefix,
        )

    def plot_top_rules_confidence(
        self,
        rules_df: pd.DataFrame,
        top_n: int = 20,
        title_prefix: str = "Top luật theo Confidence (Apriori)",
    ):
        """
        Vẽ biểu đồ top luật theo chỉ số Confidence (tương ứng code gốc ở cell 17).

        Args:
            rules_df: DataFrame, thường là rules_filtered_ap.
            top_n: số luật lấy top theo confidence.
            title_prefix: phần tiêu đề, sẽ gắn thêm số luật thực tế.
        """
        if rules_df is None or rules_df.empty:
            print("Không có luật nào sau khi lọc để vẽ top confidence.")
            return

        self.plot_top_rules_bar(
            rules_df=rules_df,
            top_n=top_n,
            sort_by="confidence",
            title=title_prefix,
        )

    def plot_rules_support_confidence_scatter(
        self,
        rules_df: pd.DataFrame,
        title: str = "Phân bố luật: Support vs Confidence (màu = Lift)",
        point_size: int = 40,
    ):
        """
        Vẽ scatter plot Support–Confidence, màu theo Lift

        Args:
            rules_df: DataFrame, thường là rules_filtered_ap.
            title: tiêu đề biểu đồ.
            point_size: kích thước điểm (tham số s của matplotlib).
        """
        if rules_df is None or rules_df.empty:
            print("Không có luật nào sau khi lọc để vẽ scatter.")
            return

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            rules_df["support"],
            rules_df["confidence"],
            c=rules_df["lift"],
            s=point_size,
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Lift")
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_pairwise_lift_heatmap(
        self,
        rules_df: pd.DataFrame,
        top_items: int = 15,
        metric: str = "lift",
        title: str = "Heatmap lift giữa các cặp sản phẩm (1→1)",
    ):
        """
        Vẽ heatmap lift (hoặc metric khác) cho các luật 1 sản phẩm → 1 sản phẩm.

        Args:
            rules_df: DataFrame kết quả từ association_rules() + add_readable_rule_str().
            top_items: số lượng sản phẩm phổ biến nhất xét đến (theo tần suất xuất hiện trong luật).
            metric: tên cột để vẽ (thường là 'lift' hoặc 'confidence').
            title: tiêu đề biểu đồ.
        """
        required_cols = {"antecedents", "consequents", metric}
        if not required_cols.issubset(set(rules_df.columns)):
            raise ValueError(f"rules_df cần có các cột: {required_cols}")

        # Chỉ giữ các luật 1 sản phẩm → 1 sản phẩm
        single_rules = rules_df[
            (rules_df["antecedents"].apply(len) == 1)
            & (rules_df["consequents"].apply(len) == 1)
        ].copy()

        if single_rules.empty:
            print("Không có luật 1→1 nào để vẽ heatmap.")
            return

        # Tạo tên sản phẩm dạng chuỗi
        single_rules["antecedent_str"] = single_rules["antecedents"].apply(
            lambda x: list(x)[0]
        )
        single_rules["consequent_str"] = single_rules["consequents"].apply(
            lambda x: list(x)[0]
        )

        # Lấy top_items sản phẩm xuất hiện nhiều nhất trong luật
        all_items = pd.concat(
            [single_rules["antecedent_str"], single_rules["consequent_str"]]
        )
        top_item_names = all_items.value_counts().head(top_items).index

        df_filtered = single_rules[
            single_rules["antecedent_str"].isin(top_item_names)
            & single_rules["consequent_str"].isin(top_item_names)
        ]

        if df_filtered.empty:
            print("Sau khi lọc top_items, không còn luật nào để vẽ heatmap.")
            return

        pivot = df_filtered.pivot_table(
            index="antecedent_str",
            columns="consequent_str",
            values=metric,
            aggfunc="max",
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            linewidths=0.5,
        )
        plt.title(title + f" (metric = {metric})")
        plt.xlabel("Consequent")
        plt.ylabel("Antecedent")
        plt.tight_layout()
        plt.show()
    def plot_rules_support_confidence_scatter_interactive(
        self,
        rules_df: pd.DataFrame,
        title: str = "Biểu đồ tương tác: Support vs Confidence (màu & kích thước = Lift)"
    ):
        """
        Biểu đồ scatter tương tác bằng Plotly:
        - Trục X: support
        - Trục Y: confidence
        - Màu & kích thước điểm: lift
        - hover hiển thị rule_str

        """
        if rules_df is None or rules_df.empty:
            print("Không có luật nào sau khi lọc để vẽ scatter Plotly.")
            return

        # Đảm bảo có rule_str (nếu chưa thì gợi ý)
        if "rule_str" not in rules_df.columns:
            print("rules_df chưa có cột 'rule_str'. Hãy gọi miner.add_readable_rule_str() trước.")
            return

        fig = px.scatter(
            rules_df,
            x="support",
            y="confidence",
            color="lift",
            size="lift",
            hover_name="rule_str",
            title=title,
            labels={
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift",
            },
        )
        fig.show()

    def plot_rules_network(
        self,
        rules_df: pd.DataFrame,
        max_rules: int | None = 100,
        min_lift: float | None = None,
        title: str = "Mạng lưới các luật kết hợp (Arrow: antecedent → consequent)",
        figsize: tuple = (12, 8),
    ):
        """
        Vẽ network graph các luật kết hợp bằng networkx:
        - Node: sản phẩm
        - Edge có hướng: antecedent -> consequent
        - Độ dày cạnh tỷ lệ với lift

        """
        if rules_df is None or rules_df.empty:
            print("Không có luật nào sau khi lọc để vẽ network graph.")
            return

        required_cols = {"antecedents", "consequents", "lift"}
        if not required_cols.issubset(rules_df.columns):
            raise ValueError(f"rules_df cần có các cột: {required_cols}")

        # Lọc theo lift nếu có
        df = rules_df.copy()
        if min_lift is not None:
            df = df[df["lift"] >= min_lift]

        if df.empty:
            print("Không còn luật nào sau khi lọc theo min_lift để vẽ network graph.")
            return

        # Giới hạn số luật để network không quá rối
        if max_rules is not None:
            df = df.sort_values("lift", ascending=False).head(max_rules)

        G = nx.DiGraph()

        # Tạo node + edge
        edges = []
        for _, row in df.iterrows():
            antecedents = list(row["antecedents"])
            consequents = list(row["consequents"])
            lift_value = row["lift"]

            for a in antecedents:
                for c in consequents:
                    G.add_node(a)
                    G.add_node(c)
                    G.add_edge(a, c, weight=lift_value)
                    edges.append((a, c, lift_value))

        if not edges:
            print("Không tạo được cạnh nào cho network graph.")
            return

        # Layout
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        # Tính độ dày cạnh
        weights = [w for (_, _, w) in edges]
        max_w = max(weights)
        norm_widths = [w / max_w * 2 for w in weights]  # scale về khoảng [0, 2]

        # Vẽ node
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue")
        # Vẽ label
        nx.draw_networkx_labels(G, pos, font_size=9)

        # Vẽ edge có hướng
        nx.draw_networkx_edges(
            G,
            pos,
            arrowstyle="->",
            arrowsize=15,
            width=norm_widths,
            edge_color="gray",
        )

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# =========================================================
# 7. RULE-BASED CUSTOMER CLUSTERING (ASSOCIATION RULES -> KMEANS)
# =========================================================
class RuleBasedCustomerClusterer:
    """Tạo đặc trưng (feature) từ LUẬT KẾT HỢP, sau đó phân cụm khách hàng.

    Ý tưởng:
    - Từ dữ liệu giao dịch đã làm sạch, tạo Customer × Item (boolean) = khách đã từng mua item đó hay chưa.
    - Từ bảng luật kết hợp (rules_apriori_filtered.csv), chọn Top-K luật (theo lift/confidence/...)
    - Với mỗi khách hàng và mỗi luật, tạo feature:
        feature_j = 1 nếu khách đã mua *đủ* antecedents của luật j (tức antecedents ⊆ basket của khách)
        (có thể nhân trọng số theo lift/confidence/support)
    - (Tuỳ chọn) Ghép thêm RFM để phân cụm ổn định hơn.

    Lưu ý: Đây là cách “dùng luật để tạo embedding thô” cho khách hàng.
    """

    def __init__(
        self,
        df_clean: pd.DataFrame,
        customer_col: str = "CustomerID",
        invoice_col: str = "InvoiceNo",
        item_col: str = "Description",
        quantity_col: str = "Quantity",
        price_col: str = "UnitPrice",
        date_col: str = "InvoiceDate",
    ):
        self.df = df_clean.copy()
        self.customer_col = customer_col
        self.invoice_col = invoice_col
        self.item_col = item_col
        self.quantity_col = quantity_col
        self.price_col = price_col
        self.date_col = date_col

        # runtime artifacts
        self.customer_item_bool: pd.DataFrame | None = None
        self.customers_: list[str] | None = None
        self.rules_df_: pd.DataFrame | None = None
        self.X_: np.ndarray | None = None
        self.model_: KMeans | None = None

    @staticmethod
    def _parse_items(items_str: str) -> list[str]:
        if items_str is None:
            return []
        s = str(items_str).strip()
        if not s:
            return []
        # format mặc định trong project: "A, B, C"
        return [x.strip() for x in s.split(",") if x.strip()]

    def build_customer_item_matrix(self, threshold: int = 1) -> pd.DataFrame:
        """Tạo Customer × Item boolean (khách đã từng mua item hay chưa)."""
        df = self.df.copy()

        if self.customer_col not in df.columns:
            raise ValueError(f"Thiếu cột {self.customer_col} trong df_clean.")
        if self.item_col not in df.columns:
            raise ValueError(f"Thiếu cột {self.item_col} trong df_clean.")
        if self.quantity_col not in df.columns:
            raise ValueError(f"Thiếu cột {self.quantity_col} trong df_clean.")

        # Chuẩn hoá CustomerID thành string để tránh float ".0"
        df[self.customer_col] = (
            df[self.customer_col].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
        )

        customer_item_qty = (
            df.groupby([self.customer_col, self.item_col])[self.quantity_col]
            .sum()
            .unstack(fill_value=0)
        )
        customer_item_bool = (customer_item_qty >= threshold)
        self.customer_item_bool = customer_item_bool
        self.customers_ = customer_item_bool.index.astype(str).tolist()
        return self.customer_item_bool

    def load_rules(
        self,
        rules_csv_path: str,
        top_k: int = 200,
        sort_by: str = "lift",
        min_support: float | None = None,
        min_confidence: float | None = None,
        min_lift: float | None = None,
    ) -> pd.DataFrame:
        """Đọc rules CSV và chọn Top-K luật để tạo feature."""
        rules = pd.read_csv(rules_csv_path)

        # kỳ vọng notebook Apriori đã add_readable_rule_str()
        required_cols = {"antecedents_str", "consequents_str"}
        if not required_cols.issubset(set(rules.columns)):
            raise ValueError(
                "rules_csv_path cần có cột antecedents_str và consequents_str. "
                "Hãy đảm bảo notebook Apriori đã gọi add_readable_rule_str() và lưu lại."
            )

        # filter ngưỡng (nếu muốn “lọc lần 2”)
        if (min_support is not None) and ("support" in rules.columns):
            rules = rules[rules["support"] >= min_support]
        if (min_confidence is not None) and ("confidence" in rules.columns):
            rules = rules[rules["confidence"] >= min_confidence]
        if (min_lift is not None) and ("lift" in rules.columns):
            rules = rules[rules["lift"] >= min_lift]

        if sort_by in rules.columns:
            rules = rules.sort_values(sort_by, ascending=False)

        if top_k is not None:
            rules = rules.head(int(top_k))

        rules = rules.reset_index(drop=True)
        self.rules_df_ = rules
        return rules

    def build_rule_feature_matrix(
        self,
        weighting: str = "none",
        min_antecedent_len: int = 1,
    ) -> np.ndarray:
        """Tạo ma trận đặc trưng Customer × Rule.

        weighting:
        - "none": feature 0/1
        - "lift" / "confidence" / "support": nhân trọng số theo cột tương ứng (nếu có)
        - "lift_x_conf": lift * confidence (nếu có)
        """
        if self.customer_item_bool is None:
            self.build_customer_item_matrix()

        if self.rules_df_ is None:
            raise ValueError("Chưa load rules. Hãy gọi load_rules() trước.")

        customer_item = self.customer_item_bool
        rules = self.rules_df_.copy()

        n_customers = customer_item.shape[0]
        n_rules = rules.shape[0]
        X = np.zeros((n_customers, n_rules), dtype=np.float32)

        # map cột item -> idx để kiểm tra nhanh
        item_cols = set(customer_item.columns.astype(str))

        for j, row in rules.iterrows():
            ants = self._parse_items(row.get("antecedents_str", ""))
            if len(ants) < min_antecedent_len:
                continue

            # nếu antecedents có item không nằm trong customer_item matrix => bỏ luật (tránh KeyError)
            if any(a not in item_cols for a in ants):
                continue

            # khách hàng nào mua đủ antecedents?
            mask = customer_item[ants].all(axis=1).astype(np.float32).values

            # trọng số
            w = 1.0
            if weighting == "lift" and "lift" in row:
                w = float(row["lift"])
            elif weighting == "confidence" and "confidence" in row:
                w = float(row["confidence"])
            elif weighting == "support" and "support" in row:
                w = float(row["support"])
            elif weighting == "lift_x_conf" and ("lift" in row) and ("confidence" in row):
                w = float(row["lift"]) * float(row["confidence"])

            X[:, j] = mask * w

        self.X_ = X
        return X

    def compute_rfm(self, snapshot_date=None) -> pd.DataFrame:
        """Tính RFM trực tiếp từ df_clean (tương tự DataCleaner.compute_rfm)."""
        df = self.df.copy()
        if "TotalPrice" not in df.columns:
            df["TotalPrice"] = df[self.quantity_col] * df[self.price_col]

        df[self.customer_col] = (
            df[self.customer_col].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
        )

        if snapshot_date is None:
            snapshot_date = pd.to_datetime(df[self.date_col]).max() + pd.Timedelta(days=1)
        else:
            snapshot_date = pd.to_datetime(snapshot_date)

        rfm = df.groupby(self.customer_col).agg(
            Recency=(self.date_col, lambda x: (snapshot_date - pd.to_datetime(x).max()).days),
            Frequency=(self.invoice_col, "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        return rfm.reset_index()

    def build_final_features(
        self,
        weighting: str = "none",
        use_rfm: bool = True,
        rfm_scale: bool = True,
        rule_scale: bool = False,
        min_antecedent_len: int = 1,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Trả về (X, meta_df) với meta_df gồm CustomerID và (tuỳ chọn) RFM."""
        if self.customer_item_bool is None:
            self.build_customer_item_matrix()

        # Rule features
        X_rules = self.build_rule_feature_matrix(
            weighting=weighting,
            min_antecedent_len=min_antecedent_len,
        )

        meta = pd.DataFrame({self.customer_col: self.customers_})

        if not use_rfm:
            self.X_ = X_rules
            return X_rules, meta

        rfm = self.compute_rfm()
        meta = meta.merge(rfm, on=self.customer_col, how="left")

        X = X_rules
        # scale RFM (khuyên dùng)
        rfm_cols = ["Recency", "Frequency", "Monetary"]
        rfm_values = meta[rfm_cols].fillna(0).values.astype(np.float32)

        if rfm_scale:
            rfm_values = StandardScaler().fit_transform(rfm_values)

        if rule_scale:
            X = StandardScaler().fit_transform(X_rules)

        X_final = np.hstack([X, rfm_values]).astype(np.float32)
        self.X_ = X_final
        return X_final, meta

    @staticmethod
    def choose_k_by_silhouette(
        X: np.ndarray,
        k_min: int = 2,
        k_max: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Chọn k dựa trên silhouette score."""
        rows = []
        for k in range(int(k_min), int(k_max) + 1):
            km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            rows.append({"k": k, "silhouette": score})
        return pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)

    def fit_kmeans(
        self,
        X: np.ndarray,
        n_clusters: int,
        random_state: int = 42,
    ) -> np.ndarray:
        """Fit KMeans và trả về labels."""
        self.model_ = KMeans(n_clusters=int(n_clusters), n_init="auto", random_state=random_state)
        labels = self.model_.fit_predict(X)
        return labels

    @staticmethod
    def project_2d(X: np.ndarray, method: str = "pca", random_state: int = 42) -> np.ndarray:
        """Giảm chiều xuống 2D để vẽ."""
        method = method.lower()
        if method == "pca":
            return PCA(n_components=2, random_state=random_state).fit_transform(X)
        if method in ("svd", "truncatedsvd"):
            return TruncatedSVD(n_components=2, random_state=random_state).fit_transform(X)
        raise ValueError("method phải là 'pca' hoặc 'svd'.")
