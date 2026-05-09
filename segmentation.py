import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Customer Segmentation App made by MUKESH",
    page_icon="📊",
    layout="centered"
)

# Load Models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Sidebar
st.sidebar.title("📊 Customer Segmentation")
st.sidebar.write("AI/ML Streamlit Project")

# Title
st.title("Customer Segmentation App made by MUKESH")
st.write("Enter customer details to predict the segment")

# Inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

income = st.number_input(
    "Income",
    min_value=0,
    max_value=200000,
    value=50000
)

total_spending = st.number_input(
    "Total Spending (Sum of Purchases)",
    min_value=0,
    max_value=5000,
    value=1000
)

num_web_purchases = st.number_input(
    "Number of Web Purchases",
    min_value=0,
    max_value=100,
    value=10
)

num_store_purchases = st.number_input(
    "Number of Store Purchases",
    min_value=0,
    max_value=100,
    value=10
)

num_web_visits = st.number_input(
    "Number of Web Visits per Month",
    min_value=0,
    max_value=50,
    value=3
)

recency = st.number_input(
    "Recency (Days Since Last Purchase)",
    min_value=0,
    max_value=365,
    value=30
)

# Create Input DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

# Predict Button
if st.button("Predict Segment"):

    try:
        # Scale Input
        input_scaled = scaler.transform(input_data)

        # Predict Cluster
        with st.spinner("Predicting Segment..."):
            cluster = kmeans.predict(input_scaled)[0]

        # Cluster Names
        cluster_names = {
            0: "High Value Customers",
            1: "Regular Customers",
            2: "Low Spending Customers",
            3: "Premium Customers"
        }

        # Result
        st.success(f"Predicted Segment: Cluster {cluster}")

        st.info(
            f"Customer Type: {cluster_names.get(cluster, 'Unknown Customer')}"
        )

        # Balloons
        st.balloons()

        # Customer Summary
        st.subheader("Customer Summary")

        st.write(f"Age: {age}")
        st.write(f"Income: ₹{income}")
        st.write(f"Total Spending: ₹{total_spending}")
        st.write(f"Web Purchases: {num_web_purchases}")
        st.write(f"Store Purchases: {num_store_purchases}")
        st.write(f"Web Visits: {num_web_visits}")
        st.write(f"Recency: {recency} days")

        # Bar Chart
        chart_data = pd.DataFrame({
            "Features": [
                "Income",
                "Spending",
                "Web Purchases",
                "Store Purchases"
            ],
            "Values": [
                income,
                total_spending,
                num_web_purchases,
                num_store_purchases
            ]
        })

        st.subheader("Customer Analysis Chart")

        st.bar_chart(chart_data.set_index("Features"))

        # Pie Chart
        st.subheader("Income vs Spending")

        fig, ax = plt.subplots()

        ax.pie(
            [income, total_spending],
            labels=["Income", "Spending"],
            autopct="%1.1f%%"
        )

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
