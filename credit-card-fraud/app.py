import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset.csv")

# Load model
model = joblib.load("fraud_model.pkl")

# Page configuration
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Credit Card Fraud Detection Dashboard")

st.write("Machine Learning model to detect fraudulent credit card transactions.")

# Sidebar
st.sidebar.header("Options")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Transaction Prediction", "Dataset Insights"]
)

# ---------------- DASHBOARD ---------------- #

if menu == "Dashboard":

    st.subheader("📊 Fraud Statistics")

    col1, col2, col3 = st.columns(3)

    total = len(df)
    fraud = df["Class"].sum()
    genuine = total - fraud

    col1.metric("Total Transactions", total)
    col2.metric("Fraud Transactions", fraud)
    col3.metric("Genuine Transactions", genuine)

    st.write("")

    # Fraud distribution graph
    st.subheader("Fraud vs Genuine Distribution")

    fig, ax = plt.subplots()

    sns.countplot(x="Class", data=df, ax=ax)
    ax.set_xticklabels(["Genuine", "Fraud"])

    st.pyplot(fig)

    # Transaction amount graph
    st.subheader("Transaction Amount Distribution")

    fig2, ax2 = plt.subplots()

    sns.histplot(df["Amount"], bins=50, ax=ax2)

    st.pyplot(fig2)

# ---------------- PREDICTION ---------------- #

elif menu == "Transaction Prediction":

    st.subheader("🔍 Predict Transaction")

    st.write("Click the button to auto-fill a random transaction from the dataset.")

    if st.button("Auto Fill Transaction"):
        sample = df.sample(1)

        st.write("Sample Transaction Data")

        st.dataframe(sample)

        X_sample = sample.drop("Class", axis=1)

        prediction = model.predict(X_sample)

        if prediction[0] == 1:
            st.error("⚠️ Fraudulent Transaction Detected")
        else:
            st.success("✅ Genuine Transaction")

# ---------------- DATASET INSIGHTS ---------------- #

elif menu == "Dataset Insights":

    st.subheader("Dataset Overview")

    st.dataframe(df.head())

    st.subheader("Dataset Information")

    st.write("Shape:", df.shape)

    st.subheader("Statistical Summary")

    st.write(df.describe())

    st.subheader("Correlation Heatmap")

    fig3, ax3 = plt.subplots(figsize=(10,6))

    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax3)

    st.pyplot(fig3)