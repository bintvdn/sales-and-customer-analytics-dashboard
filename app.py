import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

# Page Config
st.set_page_config(
    page_title="Sales Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# Helper Functions
@st.cache_data
def load_data():
    df = pd.read_csv("all_data.csv")

    datetime_columns = ["order_date", "delivery_date"]
    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.sort_values("order_date").reset_index(drop=True)
    return df


def create_daily_orders_df(df):
    daily_df = (
        df.resample(rule="D", on="order_date")
        .agg({
            "order_id": "nunique",
            "total_price": "sum"
        })
        .reset_index()
        .rename(columns={
            "order_id": "order_count",
            "total_price": "revenue"
        })
    )
    return daily_df


def create_product_df(df):
    product_df = (
        df.groupby("product_name", as_index=False)
        .agg({
            "quantity_x": "sum",
            "total_price": "sum"
        })
        .rename(columns={
            "quantity_x": "units_sold",
            "total_price": "revenue"
        })
        .sort_values(by="units_sold", ascending=False)
    )
    return product_df


def create_gender_df(df):
    gender_df = (
        df.groupby("gender", as_index=False)["customer_id"]
        .nunique()
        .rename(columns={"customer_id": "customer_count"})
        .sort_values(by="customer_count", ascending=False)
    )
    return gender_df


def create_state_df(df):
    state_df = (
        df.groupby("state", as_index=False)["customer_id"]
        .nunique()
        .rename(columns={"customer_id": "customer_count"})
        .sort_values(by="customer_count", ascending=False)
    )
    return state_df


def create_rfm_df(df):
    rfm_df = (
        df.groupby("customer_id", as_index=False)
        .agg({
            "order_date": "max",
            "order_id": "nunique",
            "total_price": "sum"
        })
    )
    rfm_df.columns = ["customer_id", "last_order_date", "frequency", "monetary"]

    recent_date = df["order_date"].max().date()
    rfm_df["last_order_date"] = rfm_df["last_order_date"].dt.date
    rfm_df["recency"] = rfm_df["last_order_date"].apply(lambda x: (recent_date - x).days)

    return rfm_df.drop(columns="last_order_date")


# Load Data
all_df = load_data()

# Sidebar Filters
st.sidebar.title("Filter Panel")

min_date = all_df["order_date"].min().date()
max_date = all_df["order_date"].max().date()

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

main_df = all_df[all_df["order_date"].between(start_date, end_date)].copy()

state_options = ["All"] + sorted(main_df["state"].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("Select State", state_options)

if selected_state != "All":
    main_df = main_df[main_df["state"] == selected_state]

# Create Summary Tables
daily_orders_df = create_daily_orders_df(main_df)
product_df = create_product_df(main_df)
gender_df = create_gender_df(main_df)
state_df = create_state_df(main_df)
rfm_df = create_rfm_df(main_df)

# KPI
total_orders = daily_orders_df["order_count"].sum()
total_revenue = daily_orders_df["revenue"].sum()
total_customers = main_df["customer_id"].nunique()
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

top_product = product_df.iloc[0]["product_name"] if not product_df.empty else "-"
top_state = state_df.iloc[0]["state"] if not state_df.empty else "-"
top_gender = gender_df.iloc[0]["gender"] if not gender_df.empty else "-"

# Header
st.title("Sales Performance & Customer Insights")
st.markdown(
    "An interactive dashboard to monitor **sales trends, product performance, and customer behavior**."
)

# KPI Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Total Orders", f"{total_orders:,}")

with kpi2:
    st.metric("Total Revenue", format_currency(total_revenue, "AUD", locale="en_AU"))

with kpi3:
    st.metric("Unique Customers", f"{total_customers:,}")

with kpi4:
    st.metric("Avg Order Value", format_currency(avg_order_value, "AUD", locale="en_AU"))

st.markdown("---")

# Main Charts - Compact Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Daily Order Trend")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=daily_orders_df,
        x="order_date",
        y="order_count",
        marker="o",
        ax=ax
    )
    ax.set_xlabel("")
    ax.set_ylabel("Orders")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.subheader("Top 5 Products")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=product_df.head(5),
        x="units_sold",
        y="product_name",
        ax=ax
    )
    ax.set_xlabel("Units Sold")
    ax.set_ylabel("")
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Customer by Gender")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=gender_df,
        x="gender",
        y="customer_count",
        ax=ax
    )
    ax.set_xlabel("")
    ax.set_ylabel("Customers")
    st.pyplot(fig)

with col4:
    st.subheader("Top States by Customers")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=state_df.head(5),
        x="customer_count",
        y="state",
        ax=ax
    )
    ax.set_xlabel("Customers")
    ax.set_ylabel("")
    st.pyplot(fig)

# Bottom Summary Section
st.markdown("---")
st.subheader("Key Business Highlights")

info1, info2, info3 = st.columns(3)

with info1:
    st.info(
        f"""
        **Best-Selling Product**  
        {top_product}
        """
    )

with info2:
    st.info(
        f"""
        **Top Customer Region**  
        {top_state}
        """
    )

with info3:
    st.info(
        f"""
        **Largest Gender Segment**  
        {top_gender}
        """
    )

# RFM Snapshot
st.subheader("Customer Value Snapshot (RFM)")

rfm1, rfm2, rfm3 = st.columns(3)

with rfm1:
    st.metric("Avg Recency (Days)", round(rfm_df["recency"].mean(), 1))

with rfm2:
    st.metric("Avg Frequency", round(rfm_df["frequency"].mean(), 2))

with rfm3:
    st.metric("Avg Monetary", format_currency(rfm_df["monetary"].mean(), "AUD", locale="en_AU"))

st.caption("Built by Bintang Vandini| Retail Sales and Customer Analytics )
